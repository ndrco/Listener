import unittest

from speaker.app import SpeakerService
from speaker.config import SpeakerConfig


class FakeGateway:
    def __init__(self, events=None, *, history_text="Final **answer**."):
        self.closed = False
        self.history_calls = 0
        self.history_text = history_text
        self._events = events or [
            {
                "type": "event",
                "event": "chat",
                "payload": {"state": "final", "sessionKey": "main", "runId": "run-final"},
            }
        ]

    async def connect(self):
        return None

    async def close(self):
        self.closed = True

    async def request(self, method, params, timeout_s=10.0):
        self.history_calls += 1
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": self.history_text}],
                    "__openclaw": {"id": "reply-1"},
                }
            ]
        }

    async def events(self):
        for event in self._events:
            yield event


class RecordingSpeech:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)


class SpeakerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_chat_event_speaks_latest_history_message_once(self):
        config = SpeakerConfig()
        config.speaker.mode = "final"
        config.speaker.speak_existing_on_start = True
        gateway = FakeGateway()
        speech = RecordingSpeech()
        service = SpeakerService(config, gateway=gateway, speech=speech)

        await service.run_until_disconnect()

        self.assertTrue(gateway.closed)
        self.assertEqual(speech.spoken, ["Final answer."])

    async def test_streaming_chat_event_speaks_deltas_without_history_reads(self):
        config = SpeakerConfig()
        config.speaker.mode = "streaming"
        config.speaker.speak_existing_on_start = True
        gateway = FakeGateway(
            [
                {
                    "type": "event",
                    "event": "chat",
                    "payload": {
                        "state": "delta",
                        "sessionKey": "main",
                        "runId": "run-stream",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Первое предложение. Второе"}],
                        },
                    },
                },
                {
                    "type": "event",
                    "event": "chat",
                    "payload": {
                        "state": "final",
                        "sessionKey": "main",
                        "runId": "run-stream",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Первое предложение. Второе"}],
                        },
                    },
                },
            ]
        )
        speech = RecordingSpeech()
        service = SpeakerService(config, gateway=gateway, speech=speech)

        await service.run_until_disconnect()

        self.assertEqual(gateway.history_calls, 0)
        self.assertEqual(speech.spoken, ["Первое предложение.", "Второе"])

    async def test_streaming_ignores_other_session(self):
        config = SpeakerConfig()
        config.speaker.mode = "streaming"
        config.speaker.speak_existing_on_start = True
        gateway = FakeGateway(
            [
                {
                    "type": "event",
                    "event": "chat",
                    "payload": {
                        "state": "delta",
                        "sessionKey": "other",
                        "runId": "run-other",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Не говорить."}],
                        },
                    },
                }
            ]
        )
        speech = RecordingSpeech()
        service = SpeakerService(config, gateway=gateway, speech=speech)

        await service.run_until_disconnect()

        self.assertEqual(speech.spoken, [])

    async def test_streaming_final_without_message_falls_back_to_history(self):
        config = SpeakerConfig()
        config.speaker.mode = "streaming"
        config.speaker.speak_existing_on_start = True
        gateway = FakeGateway()
        speech = RecordingSpeech()
        service = SpeakerService(config, gateway=gateway, speech=speech)

        await service.run_until_disconnect()

        self.assertEqual(gateway.history_calls, 1)
        self.assertEqual(speech.spoken, ["Final answer."])

    async def test_streaming_final_without_message_uses_history_to_speak_missing_tail(self):
        config = SpeakerConfig()
        config.speaker.mode = "streaming"
        config.speaker.speak_existing_on_start = True
        gateway = FakeGateway(
            [
                {
                    "type": "event",
                    "event": "chat",
                    "payload": {
                        "state": "delta",
                        "sessionKey": "main",
                        "runId": "run-stream",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Первое предложение."}],
                        },
                    },
                },
                {
                    "type": "event",
                    "event": "chat",
                    "payload": {
                        "state": "final",
                        "sessionKey": "main",
                        "runId": "run-stream",
                    },
                },
            ],
            history_text="Первое предложение. Второе **предложение**.",
        )
        speech = RecordingSpeech()
        service = SpeakerService(config, gateway=gateway, speech=speech)

        await service.run_until_disconnect()

        self.assertEqual(gateway.history_calls, 1)
        self.assertEqual(speech.spoken, ["Первое предложение.", "Второе предложение."])


if __name__ == "__main__":
    unittest.main()
