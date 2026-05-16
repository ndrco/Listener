"""Emoji extraction and external display client for speaker text."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Sequence

from .config import EmojiDisplayConfig

log = logging.getLogger(__name__)

_ZWJ = "\u200d"
_KEYCAP = "\u20e3"
_TAG_CANCEL = "\U000e007f"
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?…)\]}»”’])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{«“‘])\s+")


class EmojiDisplayError(RuntimeError):
    """Raised when the external emoji display service cannot be reached."""


@dataclass(frozen=True, slots=True)
class EmojiToken:
    symbol: str
    start: int
    end: int
    name: str


@dataclass(frozen=True, slots=True)
class EmojiSpeechText:
    speech_text: str
    tokens: tuple[EmojiToken, ...]


def extract_emoji_for_speech(text: str) -> EmojiSpeechText:
    """Remove emoji from speech text and return extracted display tokens."""

    value = str(text or "")
    if not value:
        return EmojiSpeechText("", ())

    output: list[str] = []
    tokens: list[EmojiToken] = []
    index = 0
    while index < len(value):
        end = _scan_emoji_sequence(value, index)
        if end is None:
            output.append(value[index])
            index += 1
            continue

        symbol = value[index:end]
        tokens.append(EmojiToken(symbol=symbol, start=index, end=end, name=_emoji_name(symbol)))
        if output and not output[-1].isspace():
            output.append(" ")
        index = end

    if not tokens:
        return EmojiSpeechText(value.strip(), ())
    return EmojiSpeechText(_normalize_speech_text("".join(output)), tuple(tokens))


def strip_emoji_for_speech(text: str) -> str:
    return extract_emoji_for_speech(text).speech_text


class EmojiDisplayClient:
    """Small HTTP client for the external emoji-display service."""

    def __init__(self, config: EmojiDisplayConfig) -> None:
        self.config = config
        self._last_error = ""
        self._last_sent_count = 0
        self._last_sent_symbols: tuple[str, ...] = ()
        self._last_sent_at: float | None = None

    async def show_tokens(
        self,
        tokens: Sequence[EmojiToken],
        *,
        run_id: str,
        segment_id: str,
    ) -> bool:
        selected = self._select_tokens(tokens)
        if not selected or not self.config.enabled:
            return False

        if len(selected) == 1:
            path = "/v1/show"
            payload = {
                **self._token_payload(selected[0]),
                "mode": self.config.mode,
                "source": self.config.source,
                "id": _event_id(run_id, segment_id, 0),
            }
        else:
            path = "/v1/sequence"
            payload = {
                "items": [self._token_payload(token) for token in selected],
                "mode": self.config.mode,
                "source": self.config.source,
                "id": _event_id(run_id, segment_id, None),
            }

        try:
            await asyncio.to_thread(self._post_json, path, payload)
        except Exception as exc:  # noqa: BLE001 - display errors must not stop speech
            self._last_error = str(exc)
            log.warning(
                "EmojiDisplay: failed to send %d emoji(s) to %s: %s",
                len(selected),
                self.config.url,
                exc,
            )
            return False

        self._last_error = ""
        self._last_sent_count = len(selected)
        self._last_sent_symbols = tuple(token.symbol for token in selected)
        self._last_sent_at = time.time()
        log.debug(
            "EmojiDisplay: sent %d emoji(s) symbols=%s segment=%s run_id=%s",
            len(selected),
            "".join(self._last_sent_symbols),
            segment_id,
            run_id,
        )
        return True

    async def clear(self, *, reason: str) -> bool:
        if not self.config.enabled or not self.config.clear_on_interrupt:
            return False
        payload = {"source": self.config.source, "reason": str(reason or "interrupt")}
        try:
            await asyncio.to_thread(self._post_json, "/v1/clear", payload)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            log.warning("EmojiDisplay: failed to clear display at %s: %s", self.config.url, exc)
            return False
        self._last_error = ""
        self._last_sent_symbols = ()
        self._last_sent_count = 0
        self._last_sent_at = time.time()
        log.debug("EmojiDisplay: cleared display reason=%s", reason)
        return True

    def get_status(self) -> dict:
        return {
            "enabled": bool(self.config.enabled),
            "url": self.config.url,
            "send": self.config.send,
            "mode": self.config.mode,
            "clear_on_interrupt": bool(self.config.clear_on_interrupt),
            "last_error": self._last_error or None,
            "last_sent_count": self._last_sent_count,
            "last_sent_symbols": list(self._last_sent_symbols),
            "last_sent_at": self._last_sent_at,
        }

    def _select_tokens(self, tokens: Sequence[EmojiToken]) -> tuple[EmojiToken, ...]:
        if self.config.send == "none":
            return ()
        selected = tuple(tokens)
        if self.config.send == "first":
            return selected[:1]
        return selected

    def _token_payload(self, token: EmojiToken) -> dict:
        return {
            "symbol": token.symbol,
            "name": token.name,
            "hold_ms": self.config.hold_ms,
        }

    def _post_json(self, path: str, payload: dict) -> None:
        url = urllib.parse.urljoin(f"{self.config.url.rstrip('/')}/", path.lstrip("/"))
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
            "User-Agent": "listener-emoji-display/1",
        }
        if self.config.token:
            headers["Authorization"] = f"Bearer {self.config.token}"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                status = int(getattr(response, "status", response.getcode()))
                body = response.read(512)
        except urllib.error.HTTPError as exc:
            body = exc.read(512).decode("utf-8", errors="replace").strip()
            raise EmojiDisplayError(f"HTTP {exc.code}: {body or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise EmojiDisplayError(str(exc.reason)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise EmojiDisplayError(f"request timed out after {self.config.timeout_s:.2f}s") from exc

        if status < 200 or status >= 300:
            details = body.decode("utf-8", errors="replace").strip()
            raise EmojiDisplayError(f"HTTP {status}: {details}")


def _scan_emoji_sequence(text: str, index: int) -> int | None:
    end = _scan_single_emoji(text, index)
    if end is None:
        return None
    while end < len(text) and text[end] == _ZWJ:
        next_end = _scan_single_emoji(text, end + 1)
        if next_end is None:
            break
        end = next_end
    return end


def _scan_single_emoji(text: str, index: int) -> int | None:
    if index >= len(text):
        return None

    char = text[index]
    if char in "#*0123456789":
        end = index + 1
        if end < len(text) and _is_variation_selector(text[end]):
            end += 1
        if end < len(text) and text[end] == _KEYCAP:
            return end + 1
        return None

    if _is_regional_indicator(char):
        end = index + 1
        if end < len(text) and _is_regional_indicator(text[end]):
            return end + 1
        return end

    if not _is_emoji_base(char):
        return None

    end = index + 1
    if end < len(text) and _is_variation_selector(text[end]):
        end += 1
    while end < len(text) and _is_skin_tone_modifier(text[end]):
        end += 1
    while end < len(text) and _is_tag_char(text[end]):
        end += 1
    if end < len(text) and text[end] == _TAG_CANCEL:
        end += 1
    return end


def _is_emoji_base(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x1F000 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or codepoint in {
            0x00A9,
            0x00AE,
            0x203C,
            0x2049,
            0x2122,
            0x2139,
            0x3030,
            0x303D,
            0x3297,
            0x3299,
        }
    )


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _is_variation_selector(char: str) -> bool:
    codepoint = ord(char)
    return 0xFE00 <= codepoint <= 0xFE0F


def _is_skin_tone_modifier(char: str) -> bool:
    return 0x1F3FB <= ord(char) <= 0x1F3FF


def _is_tag_char(char: str) -> bool:
    return 0xE0020 <= ord(char) <= 0xE007E


def _emoji_name(symbol: str) -> str:
    names: list[str] = []
    for char in symbol:
        if char == _ZWJ or _is_variation_selector(char) or _is_tag_char(char):
            continue
        if char == _TAG_CANCEL:
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        names.append(name.casefold().replace(" ", "_"))
    return "+".join(names[:6]) or "emoji"


def _normalize_speech_text(value: str) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _event_id(run_id: str, segment_id: str, index: int | None) -> str:
    parts = [str(run_id or "run"), str(segment_id or "segment")]
    if index is not None:
        parts.append(str(index))
    return ":".join(parts)


__all__ = [
    "EmojiDisplayClient",
    "EmojiDisplayError",
    "EmojiSpeechText",
    "EmojiToken",
    "extract_emoji_for_speech",
    "strip_emoji_for_speech",
]
