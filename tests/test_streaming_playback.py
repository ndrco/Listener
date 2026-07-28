import asyncio
import threading

import pytest

import speaker.streaming_playback as streaming_playback
from speaker.streaming_playback import PCMStreamPlayer, StreamingPlaybackError


class FakeRawStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.writes = []

    def start(self):
        self.started = True

    def write(self, payload):
        self.writes.append(payload)

    def stop(self):
        self.stopped = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def test_pcm_player_streams_and_finishes():
    async def _run() -> None:
        streams = []

        def factory(**kwargs):
            stream = FakeRawStream(**kwargs)
            streams.append(stream)
            return stream

        player = PCMStreamPlayer(stream_factory=factory)
        await player.start(sample_rate=48000)
        await player.write(b"\x01\x00\x02\x00")
        status = player.get_status()
        await player.finish()

        assert len(streams) == 1
        assert streams[0].kwargs["samplerate"] == 48000
        assert streams[0].writes == [b"\x01\x00\x02\x00"]
        assert streams[0].stopped is True
        assert streams[0].closed is True
        assert status["bytes_written"] == 4
        assert player.get_status()["active"] is False

    asyncio.run(_run())


def test_pcm_player_abort_does_not_drain_stream():
    async def _run() -> None:
        stream = FakeRawStream()
        player = PCMStreamPlayer(stream_factory=lambda **_kwargs: stream)
        await player.start(sample_rate=24000)

        await player.abort()

        assert stream.aborted is True
        assert stream.stopped is False
        assert stream.closed is True

    asyncio.run(_run())


def test_native_pcm_player_normalizes_listener_output_volume(monkeypatch):
    async def _run() -> None:
        stream = FakeRawStream()
        calls = []

        class FakeSoundDevice:
            @staticmethod
            def RawOutputStream(**kwargs):
                calls.append(kwargs)
                return stream

        async def fake_normalize():
            return {"stream_ids": [47076], "route_keys": ["listener-route"]}

        monkeypatch.setattr(streaming_playback, "sd", FakeSoundDevice())
        monkeypatch.setattr(
            streaming_playback,
            "normalize_listener_output_volume_state",
            fake_normalize,
        )
        player = PCMStreamPlayer()

        await player.start(sample_rate=48000)

        assert calls
        assert player.get_status()["normalized_stream_ids"] == [47076]
        assert player.get_status()["normalization_error"] is None
        await player.abort()

    asyncio.run(_run())


def test_pcm_player_rejects_non_pcm16_format():
    async def _run() -> None:
        player = PCMStreamPlayer(stream_factory=FakeRawStream)
        with pytest.raises(StreamingPlaybackError, match="mono PCM16"):
            await player.start(sample_rate=24000, channels=2)

    asyncio.run(_run())


def test_abort_waits_for_active_pcm_write_before_closing_stream():
    class BlockingRawStream(FakeRawStream):
        def __init__(self):
            super().__init__()
            self.write_started = threading.Event()
            self.release_write = threading.Event()

        def write(self, payload):
            self.write_started.set()
            self.release_write.wait(timeout=1.0)

        def abort(self):
            super().abort()

    async def _run() -> None:
        stream = BlockingRawStream()
        player = PCMStreamPlayer(stream_factory=lambda **_kwargs: stream)
        await player.start(sample_rate=24000)
        write_task = asyncio.create_task(player.write(b"\x00\x00" * 24000))
        await asyncio.to_thread(stream.write_started.wait, 1.0)

        abort_task = asyncio.create_task(player.abort())
        await asyncio.sleep(0.02)
        assert stream.aborted is False

        stream.release_write.set()
        await asyncio.wait_for(write_task, timeout=0.25)
        await asyncio.wait_for(abort_task, timeout=0.25)

        assert stream.aborted is True
        assert player.get_status()["active"] is False

    asyncio.run(_run())
