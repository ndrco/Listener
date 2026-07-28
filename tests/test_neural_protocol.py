import io
import struct

import pytest

from speaker.neural_protocol import (
    FrameKind,
    ProtocolError,
    encode_audio,
    encode_command,
    encode_metadata,
    read_command,
    read_output_frame,
)


def test_command_round_trip_supports_unicode():
    stream = io.BytesIO(encode_command({"command": "generate", "text": "Привет"}))

    assert read_command(stream) == {"command": "generate", "text": "Привет"}
    assert read_command(stream) is None


def test_metadata_and_audio_frames_round_trip():
    stream = io.BytesIO(
        encode_metadata({"event": "start", "sample_rate": 24000})
        + encode_audio(b"\x01\x02\x03\x04")
    )

    metadata = read_output_frame(stream)
    audio = read_output_frame(stream)

    assert metadata is not None
    assert metadata.kind is FrameKind.METADATA
    assert metadata.metadata()["sample_rate"] == 24000
    assert audio is not None
    assert audio.kind is FrameKind.AUDIO
    assert audio.payload == b"\x01\x02\x03\x04"
    assert read_output_frame(stream) is None


def test_rejects_unknown_output_frame_kind():
    stream = io.BytesIO(struct.pack("!BI", 99, 0))

    with pytest.raises(ProtocolError, match="unknown frame kind"):
        read_output_frame(stream)


def test_rejects_truncated_command():
    stream = io.BytesIO(struct.pack("!I", 10) + b"{}")

    with pytest.raises(ProtocolError, match="unexpected EOF"):
        read_command(stream)
