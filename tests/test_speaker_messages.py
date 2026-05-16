import unittest

from speaker.messages import (
    MessageDeduper,
    clean_for_speech,
    extract_assistant_message_text,
    extract_latest_assistant_text,
)


class MessageExtractionTests(unittest.TestCase):
    def test_extracts_latest_visible_assistant_text(self):
        history = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "hidden"}],
                    "__openclaw": {"id": "thinking"},
                },
                {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "tool output"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Ready **now**.",
                            "textSignature": "{\"id\":\"msg-1\"}",
                        }
                    ],
                    "__openclaw": {"id": "visible"},
                },
            ]
        }

        message = extract_latest_assistant_text(history)

        self.assertIsNotNone(message)
        self.assertEqual(message.identifier, "openclaw:visible")
        self.assertEqual(message.text, "Ready now.")

    def test_skips_no_reply_and_media_only(self):
        history = {
            "messages": [
                {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "NO_REPLY"}]},
                {"role": "assistant", "content": [{"type": "audio", "source": {}}]},
            ]
        }

        message = extract_latest_assistant_text(history)

        self.assertIsNotNone(message)
        self.assertEqual(message.text, "Hello")

    def test_clean_for_speech_removes_code_and_markdown(self):
        raw = "# Title\n- See [docs](https://example.test)\n```python\nprint('x')\n```\nUse `speaker run`."

        self.assertEqual(clean_for_speech(raw), "Title See docs Use speaker run.")

    def test_clean_for_speech_removes_partial_numbered_list_marker(self):
        self.assertEqual(clean_for_speech("Intro.\n1."), "Intro.")
        self.assertEqual(clean_for_speech("Intro.\n1. First item."), "Intro. First item.")

    def test_clean_for_speech_ignores_unclosed_code_fence_tail(self):
        raw = "Before.\n```python\nprint('still streaming')"

        self.assertEqual(clean_for_speech(raw), "Before.")

    def test_extracts_single_assistant_message_text(self):
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Streaming **answer**."}],
        }

        extracted = extract_assistant_message_text(message, fallback_identifier="run:1")

        self.assertIsNotNone(extracted)
        self.assertEqual(extracted.identifier, "run:1")
        self.assertEqual(extracted.text, "Streaming answer.")

    def test_deduper_tracks_seen_messages(self):
        history = {"messages": [{"role": "assistant", "content": "Hello", "__openclaw": {"id": "a"}}]}
        message = extract_latest_assistant_text(history)
        deduper = MessageDeduper()

        self.assertFalse(deduper.seen(message))
        deduper.mark_seen(message)
        self.assertTrue(deduper.seen(message))


if __name__ == "__main__":
    unittest.main()
