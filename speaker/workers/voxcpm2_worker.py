#!/usr/bin/env python3
"""Persistent VoxCPM2 worker for the shared or an explicitly isolated environment."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speaker.neural_protocol import PROTOCOL_VERSION  # noqa: E402
from speaker.workers.runtime import WorkerRuntime  # noqa: E402


class VoxCPM2Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = WorkerRuntime(redirect_stdout=True)
        self.model: Any | None = None
        self.prompt_cache: dict | None = None
        self.sample_rate = 0
        self.np: Any | None = None
        self.torch: Any | None = None

    def run(self) -> int:
        self.runtime.start()
        started = time.perf_counter()
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - startup failure is sent to the parent
            self.runtime.metadata(
                event="error",
                phase="startup",
                protocol_version=PROTOCOL_VERSION,
                backend="voxcpm2",
                error=str(exc),
            )
            return 2
        self.runtime.metadata(
            event="ready",
            protocol_version=PROTOCOL_VERSION,
            backend="voxcpm2",
            sample_rate=self.sample_rate,
            load_time_s=round(time.perf_counter() - started, 3),
            prompt_cached=self.prompt_cache is not None,
        )

        while not self.runtime.shutting_down:
            command = self.runtime.next_command()
            if command is None:
                break
            name = str(command.get("command") or "")
            if name == "generate":
                self._handle_generate(command)
            elif name == "cancel":
                # Generation polls and acknowledges active cancels itself.
                # A cancel reaching this loop is late and must stay silent so
                # the next persistent request starts on a clean frame boundary.
                continue
            elif name == "health":
                self.runtime.metadata(
                    event="health",
                    ok=True,
                    backend="voxcpm2",
                    sample_rate=self.sample_rate,
                )
            elif name == "shutdown":
                self.runtime.metadata(event="shutdown", ok=True)
                self.runtime.shutting_down = True
            else:
                self.runtime.metadata(event="error", error=f"unknown command: {name}")
        return 0

    def _load(self) -> None:
        model_path = Path(self.args.model).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"VoxCPM2 model directory not found: {model_path}")
        reference = self._reference_path()

        import numpy as np
        import torch
        from voxcpm import VoxCPM

        self.np = np
        self.torch = torch
        if self.args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in the VoxCPM2 environment")
            torch.cuda.set_device(_cuda_index(self.args.device))

        self.model = VoxCPM.from_pretrained(
            str(model_path),
            load_denoiser=bool(self.args.load_denoiser),
            optimize=bool(self.args.optimize),
            device=self.args.device,
            local_files_only=bool(self.args.local_files_only),
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)
        if reference is not None:
            self.prompt_cache = self.model.tts_model.build_prompt_cache(
                reference_wav_path=str(reference)
            )
        if self.args.warmup:
            self._seed()
            with torch.inference_mode():
                self.model.tts_model.generate_with_prompt_cache(
                    target_text="Проверка связи.",
                    prompt_cache=self.prompt_cache,
                    inference_timesteps=self.args.inference_timesteps,
                    cfg_value=self.args.cfg_value,
                    retry_badcase=False,
                )
        if self.args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _handle_generate(self, command: dict[str, Any]) -> None:
        request_id = str(command.get("request_id") or "")
        request = command.get("request")
        if not request_id or not isinstance(request, dict):
            self.runtime.metadata(
                event="error", request_id=request_id, error="invalid generate request"
            )
            return
        text = str(request.get("text") or "").strip()
        instruction = str(request.get("instruction") or "").strip()[:300]
        if not text:
            self.runtime.metadata(event="error", request_id=request_id, error="text is empty")
            return

        self.runtime.metadata(
            event="start",
            request_id=request_id,
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=2,
            encoding="pcm_s16le",
        )
        cancelled = False
        generator: Iterator[Any] | None = None
        try:
            self._seed()
            target_text = styled_text(text, instruction)
            assert self.model is not None and self.torch is not None
            with self.torch.inference_mode():
                generator = self.model.tts_model.generate_with_prompt_cache_streaming(
                    target_text=target_text,
                    prompt_cache=self.prompt_cache,
                    inference_timesteps=self.args.inference_timesteps,
                    cfg_value=self.args.cfg_value,
                    retry_badcase=False,
                )
                for wav, _tokens, _features in generator:
                    if self.runtime.poll_control(request_id):
                        cancelled = True
                        break
                    self.runtime.audio(float_to_pcm16(wav, self.np))
                    if self.runtime.poll_control(request_id):
                        cancelled = True
                        break
        except Exception as exc:  # noqa: BLE001 - one request must not kill the resident model
            self.runtime.metadata(event="error", request_id=request_id, error=str(exc))
            return
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()

        if cancelled:
            self.runtime.metadata(event="cancelled", request_id=request_id)
        elif not self.runtime.shutting_down:
            self.runtime.metadata(event="done", request_id=request_id)

    def _reference_path(self) -> Path | None:
        value = str(self.args.reference_wav or "").strip()
        if not value:
            return None
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VoxCPM2 reference WAV not found: {path}")
        return path

    def _seed(self) -> None:
        assert self.np is not None and self.torch is not None
        random.seed(self.args.seed)
        self.np.random.seed(self.args.seed)
        self.torch.manual_seed(self.args.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(self.args.seed)


def styled_text(text: str, instruction: str) -> str:
    value = str(text or "").strip()
    style = str(instruction or "").strip()
    return f"({style}){value}" if style else value


def float_to_pcm16(wav: Any, np_module: Any) -> bytes:
    value = wav.squeeze(0)
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np_module.asarray(value, dtype=np_module.float32).reshape(-1)
    array = np_module.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2", copy=False).tobytes()


def _cuda_index(device: str) -> int:
    return int(device.rsplit(":", 1)[1]) if ":" in device else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-wav", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--compile-threads", type=int, default=4)
    parser.add_argument("--optimize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-denoiser", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1" if args.local_files_only else "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1" if args.local_files_only else "0")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(max(1, args.compile_threads)))
    return VoxCPM2Worker(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
