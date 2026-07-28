"""Safe, model-neutral speech style selection from assistant emoji."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .config import SpeechStyleConfig
from .emoji import EmojiToken


@dataclass(frozen=True, slots=True)
class StyleDefinition:
    identifier: str
    instruction: str


@dataclass(frozen=True, slots=True)
class ResolvedStyle:
    style_id: str
    instruction: str
    emoji: str | None = None
    changed: bool = False


STYLE_DEFINITIONS: dict[str, StyleDefinition] = {
    "neutral": StyleDefinition("neutral", ""),
    "warm": StyleDefinition("warm", "Speak in a warm, friendly and gentle tone."),
    "cheerful": StyleDefinition("cheerful", "Speak in a cheerful, upbeat and lively tone."),
    "calm": StyleDefinition("calm", "Speak calmly, softly and at an unhurried pace."),
    "thoughtful": StyleDefinition("thoughtful", "Speak in a thoughtful, measured tone."),
    "sad": StyleDefinition("sad", "Speak in a subdued, sad and empathetic tone."),
    "firm": StyleDefinition("firm", "Speak firmly with controlled anger; do not shout."),
    "surprised": StyleDefinition("surprised", "Speak with clear but natural surprise."),
    "playful": StyleDefinition("playful", "Speak in a playful, lightly teasing tone."),
    "urgent": StyleDefinition("urgent", "Speak urgently, clearly and slightly faster."),
    "amused": StyleDefinition("amused", "Speak in an amused tone, as if holding back laughter."),
}


_EMOJI_STYLE: dict[str, str] = {
    "🙂": "warm",
    "😊": "warm",
    "❤": "warm",
    "😄": "cheerful",
    "🎉": "cheerful",
    "✨": "cheerful",
    "😌": "calm",
    "🤔": "thoughtful",
    "🧐": "thoughtful",
    "😔": "sad",
    "😢": "sad",
    "😭": "sad",
    "💔": "sad",
    "😠": "firm",
    "😡": "firm",
    "😮": "surprised",
    "😲": "surprised",
    "🤯": "surprised",
    "😏": "playful",
    "😼": "playful",
    "😉": "playful",
    "⚠": "urgent",
    "🚨": "urgent",
    "😂": "amused",
    "🤣": "amused",
}


class EmojiStyleResolver:
    """Resolve only allowlisted emoji and optionally retain style within one run."""

    def __init__(self, config: SpeechStyleConfig | None = None) -> None:
        self.config = config or SpeechStyleConfig()
        self._runs: dict[str, ResolvedStyle] = {}

    def resolve(
        self,
        text: str,
        tokens: Sequence[EmojiToken],
        *,
        run_id: str,
    ) -> ResolvedStyle:
        default = self._default_style()
        if not self.config.enabled:
            return default

        candidates = (
            self._leading_tokens(str(text or ""), tokens)
            if self.config.leading_emoji_only
            else tuple(tokens)
        )
        selected: tuple[EmojiToken, str] | None = None
        for token in candidates:
            style_id = _EMOJI_STYLE.get(_normalize_emoji(token.symbol))
            if style_id is not None:
                selected = (token, style_id)

        run_key = str(run_id or "").strip()
        if selected is None:
            if self.config.inherit_within_run and run_key in self._runs:
                return self._runs[run_key]
            return default

        token, style_id = selected
        definition = STYLE_DEFINITIONS[style_id]
        resolved = ResolvedStyle(
            style_id=definition.identifier,
            instruction=definition.instruction,
            emoji=token.symbol,
            changed=True,
        )
        if self.config.inherit_within_run and run_key:
            self._runs[run_key] = ResolvedStyle(
                style_id=resolved.style_id,
                instruction=resolved.instruction,
                emoji=resolved.emoji,
                changed=False,
            )
            self._trim_runs()
        return resolved

    def discard(self, run_id: str) -> None:
        self._runs.pop(str(run_id or "").strip(), None)

    def clear(self) -> None:
        self._runs.clear()

    def get_status(self) -> dict[str, object]:
        return {
            "enabled": bool(self.config.enabled),
            "leading_emoji_only": bool(self.config.leading_emoji_only),
            "inherit_within_run": bool(self.config.inherit_within_run),
            "active_runs": len(self._runs),
        }

    def _default_style(self) -> ResolvedStyle:
        style_id = str(self.config.default_style or "neutral").strip().casefold() or "neutral"
        definition = STYLE_DEFINITIONS.get(style_id, STYLE_DEFINITIONS["neutral"])
        return ResolvedStyle(definition.identifier, definition.instruction)

    @staticmethod
    def _leading_tokens(text: str, tokens: Sequence[EmojiToken]) -> tuple[EmojiToken, ...]:
        leading: list[EmojiToken] = []
        cursor = 0
        for token in tokens:
            if text[cursor : token.start].strip():
                break
            leading.append(token)
            cursor = token.end
        return tuple(leading)

    def _trim_runs(self) -> None:
        if len(self._runs) <= 512:
            return
        for run_id in tuple(self._runs)[:128]:
            self._runs.pop(run_id, None)


def _normalize_emoji(symbol: str) -> str:
    return str(symbol or "").replace("\ufe0f", "").replace("\ufe0e", "")


__all__ = [
    "EmojiStyleResolver",
    "ResolvedStyle",
    "STYLE_DEFINITIONS",
    "StyleDefinition",
]
