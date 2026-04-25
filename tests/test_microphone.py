import asyncio
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.microphone import MicrophoneStream
from core.config import cfg


class DummyBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, **payload):
        self.events.append((topic, payload))


class FakeInputStream:
    def __init__(
        self,
        *,
        samplerate,
        blocksize,
        channels,
        dtype,
        callback,
        device=None,
        start=True,
        **_extra,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.callback = callback
        self.started = False
        self.stopped = False
        self.closed = False
        self.auto_start = start

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def emit(self, array: np.ndarray):
        frames = array.shape[0]
        self.callback(array, frames, None, None)


class FakeSoundDeviceModule:
    def __init__(self):
        self.instances: list[FakeInputStream] = []

    def InputStream(self, **kwargs):
        stream = FakeInputStream(**kwargs)
        self.instances.append(stream)
        return stream


def test_microphone_stream_yields_data_and_events():
    module = FakeSoundDeviceModule()
    bus = DummyBus()
    captured: dict[str, object] = {}

    async def runner():
        async with MicrophoneStream(sounddevice_module=module, bus=bus, chunk_size=4) as mic:
            instance = module.instances[-1]
            captured["instance"] = instance

            assert instance.samplerate == mic.sample_rate
            assert instance.blocksize == mic.chunk_size
            assert instance.channels == mic.channels
            assert instance.dtype == "int16"
            if hasattr(instance, "auto_start"):
                assert instance.auto_start is False

            frame = np.arange(8, dtype=np.int16).reshape(-1, 1)
            instance.emit(frame)
            captured["chunk"] = await mic.__anext__()

    asyncio.run(runner())

    chunk = captured["chunk"]
    instance: FakeInputStream = captured["instance"]  # type: ignore[assignment]

    expected_bytes = np.arange(8, dtype=np.int16).reshape(-1, 1).tobytes()
    assert chunk == expected_bytes
    assert bus.events[-1][0] == cfg.events.audio.raw_frame
    assert bus.events[-1][1]["data"] == expected_bytes

    assert instance.started
    assert instance.stopped
    assert instance.closed


def test_microphone_stream_stop_iteration_after_close():
    module = FakeSoundDeviceModule()
    bus = DummyBus()

    async def runner():
        async with MicrophoneStream(sounddevice_module=module, bus=bus, chunk_size=4) as mic:
            frame = np.arange(4, dtype=np.int16).reshape(-1, 1)
            module.instances[-1].emit(frame)
            assert await mic.__anext__() == frame.tobytes()

        with pytest.raises(StopAsyncIteration):
            await mic.__anext__()

    asyncio.run(runner())
