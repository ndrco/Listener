from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
from piper.config import SynthesisConfig
from piper.voice import PiperVoice


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent Piper synthesis worker")
    parser.add_argument("--model", required=True)
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--sentence-silence", type=float, default=0.0)
    args = parser.parse_args()

    voice = PiperVoice.load(args.model)
    syn_config = SynthesisConfig(volume=float(args.volume))
    _write_json({"ready": True})

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            text = str(request.get("text") or "")
            output = Path(str(request.get("output") or ""))
            if not text.strip() or not output:
                raise ValueError("text and output are required")
            _synthesize_to_wav(
                voice,
                text,
                output,
                syn_config=syn_config,
                sentence_silence_s=max(0.0, float(args.sentence_silence)),
            )
        except Exception as exc:  # noqa: BLE001 - reported to parent process
            _write_json({"ok": False, "error": str(exc)})
            continue
        _write_json({"ok": True})
    return 0


def _synthesize_to_wav(
    voice: PiperVoice,
    text: str,
    output: Path,
    *,
    syn_config: SynthesisConfig,
    sentence_silence_s: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks = list(voice.synthesize(text, syn_config=syn_config))
    if not chunks:
        raise RuntimeError("Piper produced no audio chunks")
    first = chunks[0]
    silence = b""
    if sentence_silence_s > 0:
        samples = int(first.sample_rate * sentence_silence_s)
        if samples > 0:
            silence = np.zeros(samples, dtype=np.int16).tobytes()
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(first.sample_channels)
        wav.setsampwidth(first.sample_width)
        wav.setframerate(first.sample_rate)
        for index, chunk in enumerate(chunks):
            if index > 0 and silence:
                wav.writeframes(silence)
            wav.writeframes(chunk.audio_int16_bytes)


def _write_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
