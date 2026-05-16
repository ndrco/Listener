from __future__ import annotations

import asyncio

from speaker.config import EmojiDisplayConfig
from speaker.emoji import EmojiDisplayClient, EmojiToken, extract_emoji_for_speech


def test_extract_emoji_for_speech_strips_symbols_and_normalizes_spacing():
    parsed = extract_emoji_for_speech("Привет 🙂! Всё ок ✨")

    assert parsed.speech_text == "Привет! Всё ок"
    assert [token.symbol for token in parsed.tokens] == ["🙂", "✨"]
    assert parsed.tokens[0].name == "slightly_smiling_face"


def test_extract_emoji_for_speech_handles_common_sequences():
    parsed = extract_emoji_for_speech("Код 👩‍💻 1️⃣ 🇷🇺 ❤️")

    assert parsed.speech_text == "Код"
    assert [token.symbol for token in parsed.tokens] == ["👩‍💻", "1️⃣", "🇷🇺", "❤️"]
    assert parsed.tokens[0].start == 4


def test_extract_emoji_for_speech_leaves_plain_digits_alone():
    parsed = extract_emoji_for_speech("Версия 1.2 готова")

    assert parsed.speech_text == "Версия 1.2 готова"
    assert parsed.tokens == ()


def test_emoji_display_client_posts_sequence_payload():
    class RecordingClient(EmojiDisplayClient):
        def __init__(self) -> None:
            super().__init__(
                EmojiDisplayConfig(
                    enabled=True,
                    url="http://display.test",
                    hold_ms=900,
                    mode="queue",
                    source="test",
                )
            )
            self.posts: list[tuple[str, dict]] = []

        def _post_json(self, path: str, payload: dict) -> None:
            self.posts.append((path, payload))

    async def _runner() -> None:
        client = RecordingClient()
        await client.show_tokens(
            (
                EmojiToken("🙂", 0, 1, "slightly_smiling_face"),
                EmojiToken("✨", 2, 3, "sparkles"),
            ),
            run_id="run-1",
            segment_id="seg-1",
        )

        assert client.posts == [
            (
                "/v1/sequence",
                {
                    "items": [
                        {
                            "symbol": "🙂",
                            "name": "slightly_smiling_face",
                            "hold_ms": 900,
                        },
                        {"symbol": "✨", "name": "sparkles", "hold_ms": 900},
                    ],
                    "mode": "queue",
                    "source": "test",
                    "id": "run-1:seg-1",
                },
            )
        ]

    asyncio.run(_runner())


def test_emoji_display_client_first_mode_posts_single_payload():
    class RecordingClient(EmojiDisplayClient):
        def __init__(self) -> None:
            super().__init__(
                EmojiDisplayConfig(
                    enabled=True,
                    url="http://display.test",
                    send="first",
                )
            )
            self.posts: list[tuple[str, dict]] = []

        def _post_json(self, path: str, payload: dict) -> None:
            self.posts.append((path, payload))

    async def _runner() -> None:
        client = RecordingClient()
        await client.show_tokens(
            (
                EmojiToken("🙂", 0, 1, "slightly_smiling_face"),
                EmojiToken("✨", 2, 3, "sparkles"),
            ),
            run_id="run-1",
            segment_id="seg-1",
        )

        assert client.posts[0][0] == "/v1/show"
        assert client.posts[0][1]["symbol"] == "🙂"
        assert client.posts[0][1]["id"] == "run-1:seg-1:0"

    asyncio.run(_runner())
