from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedMessage:
    identifier: str
    text: str
    raw_text: str


class MessageDeduper:
    def __init__(self, max_items: int = 512) -> None:
        self.max_items = max_items
        self._seen: list[str] = []
        self._set: set[str] = set()

    def seen(self, message: ExtractedMessage) -> bool:
        return message.identifier in self._set

    def mark_seen(self, message: ExtractedMessage) -> None:
        if message.identifier in self._set:
            return
        self._set.add(message.identifier)
        self._seen.append(message.identifier)
        while len(self._seen) > self.max_items:
            old = self._seen.pop(0)
            self._set.discard(old)


def extract_latest_assistant_text(history: dict[str, Any]) -> ExtractedMessage | None:
    messages = history.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        extracted = extract_assistant_message_text(message)
        if extracted is not None:
            return extracted
    return None


def extract_assistant_message_text(
    message: Any,
    *,
    fallback_identifier: str | None = None,
) -> ExtractedMessage | None:
    if not isinstance(message, dict):
        return None
    if message.get("role") != "assistant":
        return None
    raw_text, signature_id = _extract_text_parts(message.get("content"))
    if _is_silent_reply(raw_text):
        return None
    cleaned = clean_for_speech(raw_text)
    if not cleaned:
        return None
    identifier = _message_identifier(message, signature_id, raw_text, fallback_identifier)
    return ExtractedMessage(identifier=identifier, text=cleaned, raw_text=raw_text)


def clean_for_speech(text: str) -> str:
    value = str(text or "")
    if not value.strip():
        return ""
    value = re.sub(r"```.*?```", " ", value, flags=re.DOTALL)
    value = re.sub(r"```.*", " ", value, flags=re.DOTALL)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"https?://\S+", " ", value)
    value = re.sub(r"^\s{0,3}#{1,6}\s*", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s{0,3}>\s?", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*[-*+](?:\s+|$)", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*\d+[.)](?:\s+|$)", "", value, flags=re.MULTILINE)
    value = re.sub(r"[*_~]{1,3}", "", value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    value = re.sub(r"\[\[[^\]]+\]\]", " ", value)
    value = re.sub(r"\bMEDIA:\s*\S+", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _extract_text_parts(content: Any) -> tuple[str, str | None]:
    if isinstance(content, str):
        return content, None
    if not isinstance(content, list):
        return "", None
    parts: list[str] = []
    signature_id: str | None = None
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text)
        if signature_id is None:
            signature_id = _signature_id(item.get("textSignature"))
    return "\n".join(parts), signature_id


def _signature_id(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        found = parsed.get("id")
        if isinstance(found, str) and found.strip():
            return found.strip()
    return None


def _is_silent_reply(value: str) -> bool:
    normalized = re.sub(r"[\s_-]+", "", str(value or "")).casefold()
    return normalized == "noreply"


def _message_identifier(
    message: dict[str, Any],
    signature_id: str | None,
    raw_text: str,
    fallback_identifier: str | None = None,
) -> str:
    meta = message.get("__openclaw")
    if isinstance(meta, dict):
        openclaw_id = meta.get("id")
        if isinstance(openclaw_id, str) and openclaw_id.strip():
            return f"openclaw:{openclaw_id.strip()}"
    if signature_id:
        return f"signature:{signature_id}"
    response_id = message.get("responseId")
    if isinstance(response_id, str) and response_id.strip():
        return f"response:{response_id.strip()}"
    if fallback_identifier:
        return fallback_identifier
    timestamp = message.get("timestamp")
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    if timestamp is not None:
        return f"hash:{timestamp}:{digest}"
    return f"hash:{digest}"
