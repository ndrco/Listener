import numpy as np

from speaker.workers.voxcpm2_worker import float_to_pcm16, styled_text


def test_voxcpm_style_instruction_is_prefixed_to_text():
    assert styled_text(" Привет. ", " Speak warmly. ") == "(Speak warmly.)Привет."
    assert styled_text("Привет.", "") == "Привет."


def test_float_audio_is_clipped_and_converted_to_pcm16():
    wav = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)

    pcm = np.frombuffer(float_to_pcm16(wav, np), dtype="<i2")

    assert pcm.tolist() == [-32767, -32767, 0, 32767, 32767]
