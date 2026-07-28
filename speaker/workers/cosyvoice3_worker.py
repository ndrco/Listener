#!/usr/bin/env python3
"""Persistent offline Fun-CosyVoice3 worker with cached reference features."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from speaker.neural_protocol import PROTOCOL_VERSION  # noqa: E402
from speaker.workers.runtime import WorkerRuntime  # noqa: E402


SYSTEM_PROMPT = "You are a helpful assistant."


class CosyVoice3Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.runtime = WorkerRuntime(redirect_stdout=True)
        self.model: Any | None = None
        self.cached_reference: dict[str, Any] | None = None
        self.sample_rate = 0
        self.np: Any | None = None
        self.torch: Any | None = None

    def run(self) -> int:
        self.runtime.start()
        started = time.perf_counter()
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001 - startup error is returned to Listener
            self.runtime.metadata(
                event="error",
                phase="startup",
                protocol_version=PROTOCOL_VERSION,
                backend="cosyvoice3",
                error=str(exc),
            )
            return 2
        self.runtime.metadata(
            event="ready",
            protocol_version=PROTOCOL_VERSION,
            backend="cosyvoice3",
            sample_rate=self.sample_rate,
            load_time_s=round(time.perf_counter() - started, 3),
            prompt_cached=self.cached_reference is not None,
            offline=bool(self.args.local_files_only),
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
                # Do not emit a second terminal frame for a late cancel.
                continue
            elif name == "health":
                self.runtime.metadata(
                    event="health",
                    ok=True,
                    backend="cosyvoice3",
                    sample_rate=self.sample_rate,
                )
            elif name == "shutdown":
                self.runtime.metadata(event="shutdown", ok=True)
                self.runtime.shutting_down = True
            else:
                self.runtime.metadata(event="error", error=f"unknown command: {name}")
        return 0

    def _load(self) -> None:
        repo = Path(self.args.repo).expanduser().resolve()
        model_path = Path(self.args.model).expanduser().resolve()
        prompt_wav = Path(self.args.prompt_wav).expanduser().resolve()
        if not (repo / "cosyvoice").is_dir():
            raise FileNotFoundError(f"CosyVoice repository not found: {repo}")
        if not (repo / "third_party" / "Matcha-TTS").is_dir():
            raise FileNotFoundError(f"CosyVoice Matcha-TTS submodule not found: {repo}")
        if not model_path.is_dir():
            raise FileNotFoundError(f"CosyVoice3 model directory not found: {model_path}")
        if not prompt_wav.is_file():
            raise FileNotFoundError(f"CosyVoice3 prompt WAV not found: {prompt_wav}")

        sys.path.insert(0, str(repo))
        sys.path.insert(0, str(repo / "third_party" / "Matcha-TTS"))
        self._configure_wetext()

        import numpy as np
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self.np = np
        self.torch = torch
        if self.args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in the CosyVoice3 environment")
            torch.cuda.set_device(_cuda_index(self.args.device))

        self.model = AutoModel(
            model_dir=str(model_path),
            load_trt=bool(self.args.load_trt),
            fp16=bool(self.args.fp16),
        )
        self.sample_rate = int(self.model.sample_rate)
        self.cached_reference = self._build_reference_cache(str(prompt_wav))
        if self.args.warmup:
            with torch.inference_mode():
                for _ in self._generate_items(
                    "Проверка связи.",
                    "Speak in a calm and natural tone.",
                    stream=False,
                ):
                    pass
        if self.args.device.startswith("cuda"):
            torch.cuda.synchronize()

    def _configure_wetext(self) -> None:
        import wetext.wetext as wetext_impl

        wetext_path = Path(self.args.wetext_path).expanduser().resolve()
        required = (
            wetext_path / "en" / "tn" / "tagger.fst",
            wetext_path / "en" / "tn" / "verbalizer.fst",
            wetext_path / "zh" / "tn" / "tagger.fst",
            wetext_path / "zh" / "tn" / "verbalizer.fst",
        )
        if all(path.is_file() for path in required):
            # Normalizer resolves snapshot_download in wetext.wetext globals,
            # not on the package facade imported as plain ``wetext``.
            wetext_impl.snapshot_download = lambda *_args, **_kwargs: str(wetext_path)
            return
        if self.args.local_files_only:
            def _offline_failure(*_args, **_kwargs):
                raise FileNotFoundError(f"local WeText FST directory not found: {wetext_path}")

            wetext_impl.snapshot_download = _offline_failure

    def _build_reference_cache(self, prompt_wav: str) -> dict[str, Any]:
        assert self.model is not None
        frontend = self.model.frontend
        cached = frontend.frontend_zero_shot(
            "",
            normalize_instruction(""),
            prompt_wav,
            self.sample_rate,
            "",
        )
        cached.pop("text", None)
        cached.pop("text_len", None)
        return cached

    def _model_input(self, text: str, instruction: str) -> dict[str, Any]:
        assert self.model is not None and self.cached_reference is not None
        frontend = self.model.frontend
        text_token, text_token_len = frontend._extract_text_token(text)
        instruction_token, instruction_token_len = frontend._extract_text_token(
            normalize_instruction(instruction)
        )
        model_input = dict(self.cached_reference)
        model_input.update(
            text=text_token,
            text_len=text_token_len,
            prompt_text=instruction_token,
            prompt_text_len=instruction_token_len,
        )
        model_input.pop("llm_prompt_speech_token", None)
        model_input.pop("llm_prompt_speech_token_len", None)
        return model_input

    def _generate_items(
        self,
        text: str,
        instruction: str,
        *,
        stream: bool,
    ) -> Iterator[dict[str, Any]]:
        assert self.model is not None
        model_input = self._model_input(text, instruction)
        return self.model.model.tts(
            **model_input,
            stream=stream,
            speed=self.args.speed,
        )

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
        style_id = str(request.get("style_id") or "neutral")
        if self.args.enable_vocal_events and style_id == "amused":
            text = f"[laughter] {text}"
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
        generator: Iterator[dict[str, Any]] | None = None
        try:
            assert self.torch is not None
            with self.torch.inference_mode():
                generator = self._generate_items(text, instruction, stream=True)
                for item in generator:
                    if self.runtime.poll_control(request_id):
                        cancelled = True
                        break
                    self.runtime.audio(float_to_pcm16(item["tts_speech"], self.np))
                    if self.runtime.poll_control(request_id):
                        cancelled = True
                        break
        except Exception as exc:  # noqa: BLE001 - retain loaded model after request failure
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


def normalize_instruction(instruction: str) -> str:
    value = str(instruction or "").strip()
    suffix = f" {value}" if value else ""
    return f"{SYSTEM_PROMPT}{suffix}<|endofprompt|>"


def float_to_pcm16(wav: Any, np_module: Any) -> bytes:
    value = wav.detach() if hasattr(wav, "detach") else wav
    value = value.cpu() if hasattr(value, "cpu") else value
    array = np_module.asarray(value, dtype=np_module.float32).reshape(-1)
    array = np_module.clip(array, -1.0, 1.0)
    return (array * 32767.0).astype("<i2", copy=False).tobytes()


def _cuda_index(device: str) -> int:
    return int(device.rsplit(":", 1)[1]) if ":" in device else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wetext-path", default="")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-trt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-vocal-events", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
    return CosyVoice3Worker(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
