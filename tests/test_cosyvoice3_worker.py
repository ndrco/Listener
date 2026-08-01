import sys
from argparse import Namespace
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

from speaker.workers.cosyvoice3_worker import (
    CosyVoice3Worker,
    float_to_pcm16,
    install_soundfile_wav_loader,
    load_wav_with_soundfile,
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


def test_cosyvoice_loads_reference_wav_via_soundfile(tmp_path):
    import soundfile as sf

    wav_path = tmp_path / "stereo.wav"
    samples = np.array([[0.5, -0.5], [0.25, 0.25]], dtype=np.float32)
    sf.write(wav_path, samples, 24000, subtype="FLOAT")

    speech = load_wav_with_soundfile(wav_path, 24000)

    assert tuple(speech.shape) == (1, 2)
    assert speech.numpy().ravel().tolist() == pytest.approx([0.0, 0.25])


def test_cosyvoice_installs_soundfile_loader_in_both_upstream_modules():
    frontend = ModuleType("cosyvoice.cli.frontend")
    file_utils = ModuleType("cosyvoice.utils.file_utils")
    frontend.load_wav = object()
    file_utils.load_wav = object()

    with patch.dict(
        sys.modules,
        {
            "cosyvoice.cli.frontend": frontend,
            "cosyvoice.utils.file_utils": file_utils,
        },
    ):
        install_soundfile_wav_loader()

    assert frontend.load_wav is load_wav_with_soundfile
    assert file_utils.load_wav is load_wav_with_soundfile


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
