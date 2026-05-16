import unittest

from speaker.config import GatewayConfig, StreamingConfig
from speaker.events import ChatSpeechRouter


def chat_event(state, text=None, *, run_id="run-1", session_key="main"):
    payload = {
        "runId": run_id,
        "sessionKey": session_key,
        "state": state,
    }
    if text is not None:
        payload["message"] = {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        }
    return {"type": "event", "event": "chat", "payload": payload}


class ChatSpeechRouterTests(unittest.TestCase):
    def test_delta_emits_completed_sentence_once(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        first = router.route(chat_event("delta", "Привет."))
        second = router.route(chat_event("delta", "Привет."))

        self.assertEqual([segment.text for segment in first.segments], ["Привет."])
        self.assertEqual(second.segments, [])

    def test_cumulative_delta_only_emits_new_sentence(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        router.route(chat_event("delta", "Привет."))
        result = router.route(chat_event("delta", "Привет. Мир."))

        self.assertEqual([segment.text for segment in result.segments], ["Мир."])

    def test_incomplete_tail_waits_until_final(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        delta = router.route(chat_event("delta", "Привет"))
        final = router.route(chat_event("final", "Привет"))

        self.assertEqual(delta.segments, [])
        self.assertEqual([segment.text for segment in final.segments], ["Привет"])
        self.assertTrue(final.segments[0].final)

    def test_final_flushes_remainder_after_streamed_sentence(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        delta = router.route(chat_event("delta", "Первое. Второе"))
        final = router.route(chat_event("final", "Первое. Второе"))

        self.assertEqual([segment.text for segment in delta.segments], ["Первое."])
        self.assertEqual([segment.text for segment in final.segments], ["Второе"])

    def test_final_without_message_flushes_existing_delta_remainder(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        router.route(chat_event("delta", "Первое. Второе"))
        final = router.route(chat_event("final"))

        self.assertEqual([segment.text for segment in final.segments], ["Второе"])
        self.assertTrue(final.needs_history)

    def test_history_final_text_speaks_missing_suffix_after_stale_delta(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        router.route(chat_event("delta", "Первое предложение."))
        final = router.route(chat_event("final"))
        history = router.route_final_text(
            "run-1",
            "Первое предложение. Второе предложение.",
        )

        self.assertEqual(final.segments, [])
        self.assertTrue(final.needs_history)
        self.assertEqual([segment.text for segment in history.segments], ["Второе предложение."])
        self.assertTrue(history.segments[0].final)

    def test_non_prefix_delta_stops_without_repeating(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        first = router.route(chat_event("delta", "Первое."))
        broken = router.route(chat_event("delta", "Другое."))
        final = router.route(chat_event("final", "Другое."))

        self.assertEqual([segment.text for segment in first.segments], ["Первое."])
        self.assertEqual(broken.segments, [])
        self.assertEqual(final.segments, [])
        self.assertFalse(final.needs_history)

    def test_partial_numbered_list_marker_does_not_break_streaming(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        first = router.route(chat_event("delta", "Вступление.\n1."))
        second = router.route(chat_event("delta", "Вступление.\n1. Первый пункт готов."))
        third = router.route(
            chat_event("delta", "Вступление.\n1. Первый пункт готов.\n2. Второй пункт готов.")
        )

        self.assertEqual([segment.text for segment in first.segments], ["Вступление."])
        self.assertEqual([segment.text for segment in second.segments], ["Первый пункт готов."])
        self.assertEqual([segment.text for segment in third.segments], ["Второй пункт готов."])

    def test_final_without_message_requests_history_fallback(self):
        router = ChatSpeechRouter(GatewayConfig(), StreamingConfig())

        result = router.route(chat_event("final"))

        self.assertEqual(result.segments, [])
        self.assertTrue(result.needs_history)

    def test_ignores_other_session(self):
        router = ChatSpeechRouter(GatewayConfig(session_key="main"), StreamingConfig())

        result = router.route(chat_event("delta", "Привет.", session_key="other"))

        self.assertEqual(result.segments, [])


if __name__ == "__main__":
    unittest.main()
