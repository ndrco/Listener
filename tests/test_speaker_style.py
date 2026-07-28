from speaker.config import SpeechStyleConfig
from speaker.emoji import extract_emoji_for_speech
from speaker.style import EmojiStyleResolver


def _resolve(resolver: EmojiStyleResolver, text: str, run_id: str = "run-1"):
    parsed = extract_emoji_for_speech(text)
    return resolver.resolve(text, parsed.tokens, run_id=run_id)


def test_leading_allowlisted_emoji_selects_style():
    style = _resolve(EmojiStyleResolver(), "😊 Привет!")

    assert style.style_id == "warm"
    assert style.emoji == "😊"
    assert "warm" in style.instruction
    assert style.changed is True


def test_pensive_face_selects_sad_style():
    style = _resolve(EmojiStyleResolver(), "😔 Иногда тишина всё понимает.")

    assert style.style_id == "sad"
    assert style.emoji == "😔"
    assert "sad" in style.instruction


def test_trailing_emoji_is_display_only_by_default():
    style = _resolve(EmojiStyleResolver(), "Привет! 😊")

    assert style.style_id == "neutral"
    assert style.emoji is None


def test_style_is_inherited_only_within_same_run():
    resolver = EmojiStyleResolver()

    _resolve(resolver, "😌 Спокойно.", "run-1")
    inherited = _resolve(resolver, "Продолжаем.", "run-1")
    other_run = _resolve(resolver, "Новый ответ.", "run-2")

    assert inherited.style_id == "calm"
    assert inherited.changed is False
    assert other_run.style_id == "neutral"


def test_unknown_emoji_never_becomes_instruction():
    style = _resolve(EmojiStyleResolver(), "🛠️ Собираю проект.")

    assert style.style_id == "neutral"
    assert style.instruction == ""
    assert style.emoji is None


def test_discard_resets_inherited_run_style():
    resolver = EmojiStyleResolver()
    _resolve(resolver, "🚨 Срочно.", "run-1")

    resolver.discard("run-1")

    assert _resolve(resolver, "Продолжение.", "run-1").style_id == "neutral"


def test_style_resolution_can_be_disabled():
    resolver = EmojiStyleResolver(SpeechStyleConfig(enabled=False))

    assert _resolve(resolver, "😄 Радостно.").style_id == "neutral"
