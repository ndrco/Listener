"""Framing shared by Listener and isolated neural TTS worker processes."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, BinaryIO


PROTOCOL_VERSION = 1
MAX_COMMAND_BYTES = 1 * 1024 * 1024
MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_AUDIO_BYTES = 16 * 1024 * 1024
_LENGTH = struct.Struct("!I")
_FRAME_HEADER = struct.Struct("!BI")


class ProtocolError(RuntimeError):
    pass


class FrameKind(IntEnum):
    METADATA = 1
    AUDIO = 2


@dataclass(frozen=True, slots=True)
class OutputFrame:
    kind: FrameKind
    payload: bytes

    def metadata(self) -> dict[str, Any]:
        if self.kind is not FrameKind.METADATA:
            raise ProtocolError("audio frame does not contain JSON metadata")
        return decode_json_object(self.payload)


def encode_command(command: dict[str, Any]) -> bytes:
    payload = _encode_json_object(command)
    if len(payload) > MAX_COMMAND_BYTES:
        raise ProtocolError(f"command exceeds {MAX_COMMAND_BYTES} bytes")
    return _LENGTH.pack(len(payload)) + payload


def read_command(stream: BinaryIO) -> dict[str, Any] | None:
    header = _read_exact_or_eof(stream, _LENGTH.size)
    if header is None:
        return None
    (length,) = _LENGTH.unpack(header)
    if length > MAX_COMMAND_BYTES:
        raise ProtocolError(f"command frame exceeds {MAX_COMMAND_BYTES} bytes")
    return decode_json_object(_read_exact(stream, length))


def encode_metadata(metadata: dict[str, Any]) -> bytes:
    payload = _encode_json_object(metadata)
    if len(payload) > MAX_METADATA_BYTES:
        raise ProtocolError(f"metadata exceeds {MAX_METADATA_BYTES} bytes")
    return _FRAME_HEADER.pack(FrameKind.METADATA, len(payload)) + payload


def encode_audio(pcm: bytes) -> bytes:
    payload = bytes(pcm)
    if len(payload) > MAX_AUDIO_BYTES:
        raise ProtocolError(f"audio chunk exceeds {MAX_AUDIO_BYTES} bytes")
    return _FRAME_HEADER.pack(FrameKind.AUDIO, len(payload)) + payload


def read_output_frame(stream: BinaryIO) -> OutputFrame | None:
    header = _read_exact_or_eof(stream, _FRAME_HEADER.size)
    if header is None:
        return None
    kind, length = decode_output_header(header)
    limit = MAX_METADATA_BYTES if kind is FrameKind.METADATA else MAX_AUDIO_BYTES
    if length > limit:
        raise ProtocolError(f"{kind.name.lower()} frame exceeds {limit} bytes")
    return OutputFrame(kind=kind, payload=_read_exact(stream, length))


def decode_output_header(header: bytes) -> tuple[FrameKind, int]:
    if len(header) != _FRAME_HEADER.size:
        raise ProtocolError(f"output header must be {_FRAME_HEADER.size} bytes")
    raw_kind, length = _FRAME_HEADER.unpack(header)
    try:
        kind = FrameKind(raw_kind)
    except ValueError as exc:
        raise ProtocolError(f"unknown frame kind: {raw_kind}") from exc
    return kind, length


def decode_json_object(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON payload: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("JSON payload must be an object")
    return value


def _encode_json_object(value: dict[str, Any]) -> bytes:
    if not isinstance(value, dict):
        raise ProtocolError("JSON payload must be an object")
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"JSON payload is not serializable: {exc}") from exc


def _read_exact_or_eof(stream: BinaryIO, size: int) -> bytes | None:
    first = stream.read(size)
    if not first:
        return None
    if len(first) == size:
        return first
    return first + _read_exact(stream, size - len(first))


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError(f"unexpected EOF with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = [
    "FrameKind",
    "MAX_AUDIO_BYTES",
    "MAX_COMMAND_BYTES",
    "MAX_METADATA_BYTES",
    "OutputFrame",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "decode_json_object",
    "decode_output_header",
    "encode_audio",
    "encode_command",
    "encode_metadata",
    "read_command",
    "read_output_frame",
]
