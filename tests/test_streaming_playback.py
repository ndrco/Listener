import asyncio
import threading

import pytest

import speaker.streaming_playback as streaming_playback
from speaker.streaming_playback import (
    PCMStreamPlayer,
    StreamingPlaybackError,
    StreamingPlaybackInterrupted,
    build_pcm_command,
    resolve_pcm_backend,
)


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


class FakeProcessStdin:
    def __init__(self, process, *, fail_drain=False):
        self.process = process
        self.fail_drain = fail_drain
        self.buffer = bytearray()
        self.closed = False

    def write(self, payload):
        self.buffer.extend(payload)

    async def drain(self):
        if self.fail_drain:
            self.process.returncode = 11
            raise BrokenPipeError("simulated player crash")

    def close(self):
        self.closed = True
        if self.process.returncode is None:
            self.process.returncode = 0

    async def wait_closed(self):
        return None


class FakeProcessStderr:
    async def read(self, _size):
        return b""


class FakeProcess:
    def __init__(self, *, pid=4321, fail_drain=False):
        self.pid = pid
        self.returncode = None
        self.stdin = FakeProcessStdin(self, fail_drain=fail_drain)
        self.stderr = FakeProcessStderr()

    async def wait(self):
        while self.returncode is None:
            await asyncio.sleep(0)
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def test_pcm_player_streams_and_finishes():
    async def _run() -> None:
        streams = []

        def factory(**kwargs):
            stream = FakeRawStream(**kwargs)
            streams.append(stream)
            return stream

        player = PCMStreamPlayer(stream_factory=factory, prebuffer_ms=0)
        await player.start(sample_rate=48000)
        await player.write(b"\x01\x00\x02\x00")
        status = player.get_status()
        await player.finish()

        assert len(streams) == 1
        assert streams[0].kwargs["samplerate"] == 48000
        assert streams[0].writes == [b"\x01\x00\x02\x00"]
        assert streams[0].stopped is True
        assert streams[0].closed is True
        assert status["bytes_enqueued"] == 4
        assert player.get_status()["last_bytes_written"] == 4
        assert player.get_status()["active"] is False

    asyncio.run(_run())


def test_pcm_player_abort_does_not_drain_stream():
    async def _run() -> None:
        stream = FakeRawStream()
        player = PCMStreamPlayer(
            stream_factory=lambda **_kwargs: stream,
            prebuffer_ms=0,
        )
        await player.start(sample_rate=24000)
        await player.write(b"\x00\x00")

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
        player = PCMStreamPlayer(backend="sounddevice", prebuffer_ms=0)

        await player.start(sample_rate=48000)
        await player.write(b"\x00\x00")

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
        player = PCMStreamPlayer(
            stream_factory=lambda **_kwargs: stream,
            prebuffer_ms=0,
        )
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


def test_subprocess_player_prebuffers_once_and_reuses_run():
    async def _run() -> None:
        processes = []
        lifecycle = []

        async def process_factory(*args, **kwargs):
            process = FakeProcess()
            process.args = args
            process.kwargs = kwargs
            processes.append(process)
            return process

        async def on_start():
            lifecycle.append("start")

        async def on_stop():
            lifecycle.append("stop")

        player = PCMStreamPlayer(
            backend="pacat",
            command="/usr/bin/pacat",
            prebuffer_ms=100,
            latency_ms=80,
            queue_ms=500,
            process_factory=process_factory,
            on_playback_start=on_start,
            on_playback_stop=on_stop,
        )
        await player.start(sample_rate=1000, run_id="run-1")
        await player.write(b"\x00\x00" * 50)
        assert processes == []
        assert lifecycle == []

        await player.write(b"\x01\x00" * 50)
        await player.finish_segment()
        await player.start(sample_rate=1000, run_id="run-1")
        await player.write(b"\x02\x00" * 50)
        await player.finish_segment()
        assert len(processes) == 1
        assert lifecycle == ["start"]

        await player.finish_run("run-1")

        assert bytes(processes[0].stdin.buffer) == (
            b"\x00\x00" * 50 + b"\x01\x00" * 50 + b"\x02\x00" * 50
        )
        assert lifecycle == ["start", "stop"]
        assert player.get_status()["active"] is False
        assert player.get_status()["last_bytes_written"] == 300

    asyncio.run(_run())


def test_subprocess_crash_is_reported_as_recoverable_playback_failure():
    async def _run() -> None:
        process = FakeProcess(fail_drain=True)

        async def process_factory(*_args, **_kwargs):
            return process

        player = PCMStreamPlayer(
            backend="pacat",
            command="/usr/bin/pacat",
            prebuffer_ms=0,
            process_factory=process_factory,
        )
        await player.start(sample_rate=24000, run_id="run-crash")
        await player.write(b"\x00\x00" * 480)
        for _ in range(100):
            if player.get_status()["last_error"]:
                break
            await asyncio.sleep(0)

        with pytest.raises(StreamingPlaybackInterrupted, match="stopped accepting PCM"):
            await player.finish_run("run-crash")
        await player.abort()

        status = player.get_status()
        assert status["active"] is False
        assert status["restart_count"] == 1

    asyncio.run(_run())


def test_pcm_backend_resolution_and_pacat_command(monkeypatch):
    monkeypatch.setattr(streaming_playback.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert resolve_pcm_backend("auto") == ("pacat", "/usr/bin/pacat")
    args = build_pcm_command(
        "pacat",
        "/usr/bin/pacat",
        sample_rate=24000,
        channels=1,
        latency_ms=100,
        client_name="Speaker",
        stream_name="Speaker TTS",
    )
    assert "--raw" in args
    assert "--rate=24000" in args
    assert "--latency-msec=100" in args
    assert "--property=application.id=speaker" in args
