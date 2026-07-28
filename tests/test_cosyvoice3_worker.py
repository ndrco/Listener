import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from speaker.workers.cosyvoice3_worker import (
    CosyVoice3Worker,
    float_to_pcm16,
    normalize_instruction,
)


def test_cosyvoice_instruction_has_required_system_delimiter():
    assert normalize_instruction("Speak warmly.") == (
        "You are a helpful assistant. Speak warmly.<|endofprompt|>"
    )
    assert normalize_instruction("") == "You are a helpful assistant.<|endofprompt|>"


def test_cosyvoice_float_audio_is_pcm16():
    wav = np.array([[-1.0, 0.0, 0.5, 1.0]], dtype=np.float32)

    pcm = np.frombuffer(float_to_pcm16(wav, np), dtype="<i2")

    assert pcm.tolist() == [-32767, 0, 16383, 32767]


def test_cosyvoice_redirects_wetext_download_to_configured_fst_directory(tmp_path):
    for relative in (
        "en/tn/tagger.fst",
        "en/tn/verbalizer.fst",
        "zh/tn/tagger.fst",
        "zh/tn/verbalizer.fst",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()

    package = ModuleType("wetext")
    implementation = ModuleType("wetext.wetext")
    implementation.snapshot_download = lambda *_args, **_kwargs: "network"
    package.wetext = implementation
    worker = CosyVoice3Worker.__new__(CosyVoice3Worker)
    worker.args = Namespace(wetext_path=str(tmp_path), local_files_only=True)

    with patch.dict(
        sys.modules,
        {"wetext": package, "wetext.wetext": implementation},
    ):
        worker._configure_wetext()

    assert implementation.snapshot_download("pengzhendong/wetext") == str(tmp_path)


def test_cosyvoice_offline_wetext_fails_without_local_fst(tmp_path):
    package = ModuleType("wetext")
    implementation = ModuleType("wetext.wetext")
    package.wetext = implementation
    worker = CosyVoice3Worker.__new__(CosyVoice3Worker)
    worker.args = Namespace(wetext_path=str(tmp_path), local_files_only=True)

    with patch.dict(
        sys.modules,
        {"wetext": package, "wetext.wetext": implementation},
    ):
        worker._configure_wetext()

    with pytest.raises(FileNotFoundError, match="local WeText FST directory"):
        implementation.snapshot_download("pengzhendong/wetext")
