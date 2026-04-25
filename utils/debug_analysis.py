"""Utilities to inspect diarizer debug artifacts.

The script walks through ``debug/audio_speaker`` runs and summarizes
waveform levels together with mel-spectrogram sparsity, so it is easy to
spot abnormally large zero areas in the extracted features.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(slots=True)
class WavStats:
    duration_sec: float
    rms: float
    peak: float
    mean: float
    std: float
    zero_fraction: float


@dataclass(slots=True)
class MelStats:
    shape: List[int]
    min: float
    max: float
    mean: float
    std: float
    zero_fraction: float
    zero_rows: int
    zero_cols: int


@dataclass(slots=True)
class RunSummary:
    run_id: str
    manifest: str | None
    raw_pcm: WavStats | None
    resampled: WavStats | None
    normalized: WavStats | None
    mel_pre: MelStats | None
    mel_post: MelStats | None


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        raw = wf.readframes(frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    data = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        data = data.reshape((-1, channels)).mean(axis=1)

    return data.astype(np.float32) / float(np.iinfo(dtype).max), sample_rate


def _wav_stats(wav: np.ndarray, sample_rate: int) -> WavStats:
    if sample_rate <= 0:
        duration = 0.0
    else:
        duration = float(len(wav)) / float(sample_rate)

    if wav.size == 0:
        return WavStats(duration, 0.0, 0.0, 0.0, 0.0, 1.0)

    rms = float(math.sqrt(float(np.mean(np.square(wav)))))
    peak = float(np.max(np.abs(wav)))
    mean = float(np.mean(wav))
    std = float(np.std(wav))
    zero_fraction = float(np.mean(wav == 0.0))

    return WavStats(duration, rms, peak, mean, std, zero_fraction)


def _mel_stats(mel: np.ndarray) -> MelStats:
    if mel.size == 0:
        shape = [int(dim) for dim in mel.shape]
        return MelStats(shape, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0)

    zero_mask = mel == 0
    zero_rows = int(np.sum(np.all(zero_mask, axis=1))) if mel.ndim == 2 else 0
    zero_cols = int(np.sum(np.all(zero_mask, axis=0))) if mel.ndim == 2 else 0

    return MelStats(
        shape=[int(dim) for dim in mel.shape],
        min=float(np.min(mel)),
        max=float(np.max(mel)),
        mean=float(np.mean(mel)),
        std=float(np.std(mel)),
        zero_fraction=float(np.mean(zero_mask)),
        zero_rows=zero_rows,
        zero_cols=zero_cols,
    )


def _find_runs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for manifest in root.rglob("manifest.json"):
        yield manifest.parent


def _safe_load_wav(path: Path) -> tuple[np.ndarray | None, int]:
    if not path.exists():
        return None, 0
    try:
        return _load_wav(path)
    except Exception:
        return None, 0


def _safe_load_mel(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def analyze_run(run_dir: Path) -> RunSummary:
    manifest_path = run_dir / "manifest.json"
    manifest_rel: str | None = str(manifest_path) if manifest_path.exists() else None

    raw_pcm, raw_sr = _safe_load_wav(run_dir / "raw_input.wav")
    resampled, resampled_sr = _safe_load_wav(run_dir / "resampled_before_norm.wav")
    normalized, normalized_sr = _safe_load_wav(run_dir / "normalized.wav")

    mel_pre = _safe_load_mel(run_dir / "mel_pre_norm.npy")
    mel_post = _safe_load_mel(run_dir / "mel_post_norm.npy")

    return RunSummary(
        run_id=run_dir.name,
        manifest=manifest_rel,
        raw_pcm=_wav_stats(raw_pcm, raw_sr) if raw_pcm is not None else None,
        resampled=_wav_stats(resampled, resampled_sr) if resampled is not None else None,
        normalized=_wav_stats(normalized, normalized_sr) if normalized is not None else None,
        mel_pre=_mel_stats(mel_pre) if mel_pre is not None else None,
        mel_post=_mel_stats(mel_post) if mel_post is not None else None,
    )


def analyze_debug_dir(root: Path, limit: int | None = None) -> list[RunSummary]:
    summaries: list[RunSummary] = []
    for idx, run_dir in enumerate(_find_runs(root)):
        if limit is not None and idx >= limit:
            break
        summaries.append(analyze_run(run_dir))
    return summaries


def _encode_summary(summary: RunSummary) -> dict:
    def encode_wav(stats: WavStats | None) -> dict | None:
        if stats is None:
            return None
        return {
            "duration_sec": stats.duration_sec,
            "rms": stats.rms,
            "peak": stats.peak,
            "mean": stats.mean,
            "std": stats.std,
            "zero_fraction": stats.zero_fraction,
        }

    def encode_mel(stats: MelStats | None) -> dict | None:
        if stats is None:
            return None
        return {
            "shape": stats.shape,
            "min": stats.min,
            "max": stats.max,
            "mean": stats.mean,
            "std": stats.std,
            "zero_fraction": stats.zero_fraction,
            "zero_rows": stats.zero_rows,
            "zero_cols": stats.zero_cols,
        }

    return {
        "run_id": summary.run_id,
        "manifest": summary.manifest,
        "raw_pcm": encode_wav(summary.raw_pcm),
        "resampled": encode_wav(summary.resampled),
        "normalized": encode_wav(summary.normalized),
        "mel_pre": encode_mel(summary.mel_pre),
        "mel_post": encode_mel(summary.mel_post),
    }


def _print_report(
    summaries: list[RunSummary], wav_zero_thr: float, mel_zero_thr: float
) -> None:
    if not summaries:
        print("No debug runs found: directory is empty or missing.")
        return

    print(f"Runs found: {len(summaries)}\n")
    for summary in summaries:
        print(f"## {summary.run_id}")
        if summary.manifest:
            print(f"manifest: {summary.manifest}")

        def fmt_wav(label: str, stats: WavStats | None) -> None:
            if stats is None:
                print(f"  {label}: unavailable")
                return
            print(
                f"  {label}: duration={stats.duration_sec:.3f}s rms={stats.rms:.6f} "
                f"peak={stats.peak:.6f} mean={stats.mean:.6f} std={stats.std:.6f} "
                f"zeros={stats.zero_fraction:.3f}"
            )

        fmt_wav("raw", summary.raw_pcm)
        fmt_wav("resampled", summary.resampled)
        fmt_wav("normalized", summary.normalized)

        def fmt_mel(label: str, stats: MelStats | None) -> None:
            if stats is None:
                print(f"  {label}: unavailable")
                return
            print(
                f"  {label}: shape={stats.shape} min={stats.min:.6f} max={stats.max:.6f} "
                f"mean={stats.mean:.6f} std={stats.std:.6f} zeros={stats.zero_fraction:.3f} "
                f"zero_rows={stats.zero_rows} zero_cols={stats.zero_cols}"
            )

        fmt_mel("mel_pre", summary.mel_pre)
        fmt_mel("mel_post", summary.mel_post)

        issues = _detect_issues(summary, wav_zero_thr, mel_zero_thr)
        if issues:
            print("  Warning: potential issues:")
            for item in issues:
                print(f"    - {item}")

        print()


def _detect_issues(summary: RunSummary, wav_zero_thr: float, mel_zero_thr: float) -> list[str]:
    issues: list[str] = []

    def wav_issue(label: str, stats: WavStats | None) -> None:
        if stats is None:
            issues.append(f"{label}: wav is missing")
            return
        if stats.zero_fraction >= wav_zero_thr:
            issues.append(f"{label}: zero fraction {stats.zero_fraction:.3f} >= {wav_zero_thr:.3f}")

    def mel_issue(label: str, stats: MelStats | None) -> None:
        if stats is None:
            issues.append(f"{label}: mel is missing")
            return
        if stats.zero_fraction >= mel_zero_thr:
            issues.append(f"{label}: zero fraction {stats.zero_fraction:.3f} >= {mel_zero_thr:.3f}")
        if stats.zero_rows:
            issues.append(f"{label}: fully zero rows {stats.zero_rows}")
        if stats.zero_cols:
            issues.append(f"{label}: fully zero columns {stats.zero_cols}")

    wav_issue("raw", summary.raw_pcm)
    wav_issue("resampled", summary.resampled)
    wav_issue("normalized", summary.normalized)
    mel_issue("mel_pre", summary.mel_pre)
    mel_issue("mel_post", summary.mel_post)

    return issues


def _print_summary_stats(
    summaries: list[RunSummary], wav_zero_thr: float, mel_zero_thr: float
) -> None:
    runs_with_issues = 0
    for summary in summaries:
        issues = _detect_issues(summary, wav_zero_thr, mel_zero_thr)
        if issues:
            runs_with_issues += 1
    if not summaries:
        return

    print("Summary:")
    print(
        f"  runs with potential issues: {runs_with_issues} of {len(summaries)} "
        f"(zero thresholds: wav >= {wav_zero_thr:.3f}, mel >= {mel_zero_thr:.3f})"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze diarizer debug audio artifacts")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("debug/audio_speaker"),
        help="Directory with dumps (default: debug/audio_speaker)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save JSON report (if omitted, print only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of runs to analyze",
    )
    parser.add_argument(
        "--wav-zero-thr",
        type=float,
        default=0.05,
        help="Zero-fraction threshold in waveform above which issues are highlighted",
    )
    parser.add_argument(
        "--mel-zero-thr",
        type=float,
        default=0.05,
        help="Zero-fraction threshold in mel-spectrogram for issue highlighting",
    )
    args = parser.parse_args(argv)

    summaries = analyze_debug_dir(args.root, limit=args.limit)
    _print_report(summaries, wav_zero_thr=args.wav_zero_thr, mel_zero_thr=args.mel_zero_thr)
    _print_summary_stats(
        summaries, wav_zero_thr=args.wav_zero_thr, mel_zero_thr=args.mel_zero_thr
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data = [_encode_summary(summary) for summary in summaries]
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
