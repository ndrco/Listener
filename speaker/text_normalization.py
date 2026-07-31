"""Narrow Russian text normalization shared by every TTS backend.

Only selected numeric and mathematical spans are changed. Text outside those
spans is copied verbatim so that model names, URLs and ordinary Latin text
cannot be transliterated as a side effect.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable


NumberNormalizer = Callable[[str], str]
TextNormalizer = Callable[[str], str]
log = logging.getLogger(__name__)

_MONTH = (
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|"
    r"сентября|октября|ноября|декабря)"
)
_MONTH_RE = re.compile(rf"^{_MONTH}$", re.IGNORECASE)
_DATE_GENITIVE_CONTEXTS = {"около", "порядка", "до", "от", "с", "со", "без", "после"}
_DATE_DATIVE_CONTEXTS = {"к", "ко"}
_DATE_PREPOSITIONAL_CONTEXTS = {"о", "об"}

_GROUPED_INTEGER = r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)"
_DATE = r"(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4})"
_CLOCK = r"(?:\d{1,2}:\d{2}(?::\d{2})?)"
_DOTTED = r"(?:\d+(?:\.\d+){1,3})"
_FRACTION = rf"(?:{_GROUPED_INTEGER}\s*/\s*\d+)"
_DECIMAL = rf"(?:{_GROUPED_INTEGER}(?:,\d+)?)"
_CORE = rf"(?:{_DATE}|{_CLOCK}|{_DOTTED}|{_FRACTION}|{_DECIMAL})"

_CONTEXT = (
    r"(?:около|примерно|порядка|до|от|с|со|без|после|более|менее|"
    r"к|ко|по|в|во|на|за|о|об)"
)
_MAGNITUDE = (
    r"(?:тыс(?:\.|яч(?:а|и|у|ей|ами|ах)?)?|"
    r"млн\.?|миллион(?:а|у|ом|е|ы|ов|ами|ах)?|"
    r"млрд\.?|миллиард(?:а|у|ом|е|ы|ов|ами|ах)?|"
    r"трлн\.?|триллион(?:а|у|ом|е|ы|ов|ами|ах)?)"
)
_UNIT = (
    r"(?:"
    rf"{_MONTH}|"
    r"г(?:\.|од(?:а|у|ом|е|ы|ов|ами|ах)?)|"
    r"руб(?:\.|л(?:ь|я|ю|ём|ем|е|и|ей|ями|ях)?)|"
    r"коп(?:\.|ейк(?:а|и|у|ой|е|ами|ах)?)|"
    r"доллар(?:а|у|ом|е|ы|ов|ами|ах)?|евро|"
    r"процент(?:а|у|ом|е|ы|ов|ами|ах)?|"
    r"километр(?:а|у|ом|е|ы|ов|ами|ах)?|км(?:/ч)?|"
    r"метр(?:а|у|ом|е|ы|ов|ами|ах)?|м(?:/с)?|см|мм|мкм|нм|"
    r"килограмм(?:а|у|ом|е|ы|ов|ами|ах)?|кг|мг|"
    r"грамм(?:а|у|ом|е|ы|ов|ами|ах)?|"
    r"литр(?:а|у|ом|е|ы|ов|ами|ах)?|л|мл|"
    r"час(?:а|у|ом|е|ы|ов|ами|ах)?|ч|"
    r"минут(?:а|ы|у|ой|е|ами|ах)?|мин\.?|"
    r"секунд(?:а|ы|у|ой|е|ами|ах)?|сек\.?|с|"
    r"мест(?:о|а|у|ом|е)|этаж(?:а|у|ом|е|и|ей|ами|ах)?|"
    r"(?:[KMGTКМГТ]?B|[КМГТ]?Б)(?:/с)?|"
    r"[кмМГ]?Гц|[кмМ]?Вт|Вт|В|А|"
    r"°\s*[CFСФ]|RUB|USD|EUR"
    r")"
)
_ORDINAL = r"(?:-(?:й|я|е|ый|ая|ое|го|му|м|ую|ых|ми))"

_NUMERIC_SPAN_RE = re.compile(
    rf"(?<!\w)"
    rf"(?P<context>{_CONTEXT}\s+)?"
    rf"(?P<symbol>[№$€₽£¥+-]\s*)?"
    rf"(?P<core>{_CORE})"
    rf"(?P<ordinal>{_ORDINAL})?"
    rf"(?P<suffix>(?:\s+{_MAGNITUDE})?(?:\s+{_UNIT})?(?:\s*%)?)"
    rf"(?!\w)",
    re.IGNORECASE,
)

_PROTECTED_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+|"
    r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
    r"`[^`]*`|<\|.*?\|>|"
    r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)|"
    r"(?<!\w)\+\d(?:[\d ()-]*\d){6,}(?!\w)|"
    r"(?<!\w)/(?:[^\s/]+/)+[^\s/]+",
    re.IGNORECASE,
)

_VERSION_WORD_RE = re.compile(
    r"(?:\b(?:верси(?:я|и|ю|ей|ях)|version|ver|v)|\b[A-Za-z][\w.-]*)\s*$",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(
    r"[$€₽£¥]|\b(?:RUB|USD|EUR|руб|доллар|евро|коп)",
    re.IGNORECASE,
)
_SINGLE_EQUALS_RE = re.compile(r"(?<![<>=!])=(?!=)")


def normalize_russian_numeric_spans(text: str, normalizer: NumberNormalizer) -> str:
    """Normalize numeric expressions while preserving unrelated text exactly."""

    value = str(text or "")
    if not any(character.isdecimal() for character in value):
        return _replace_equals(value)

    protected = tuple(match.span() for match in _PROTECTED_RE.finditer(value))
    output: list[str] = []
    cursor = 0
    for match in _NUMERIC_SPAN_RE.finditer(value):
        if _overlaps(match.span(), protected):
            continue
        output.append(value[cursor : match.start()])
        output.append(_normalize_match(value, match, normalizer))
        cursor = match.end()
    if cursor == 0:
        normalized = value
    else:
        output.append(value[cursor:])
        normalized = "".join(output)
    return _replace_equals(normalized)


def _normalize_match(
    text: str,
    match: re.Match[str],
    normalizer: NumberNormalizer,
) -> str:
    source = match.group(0)
    core = match.group("core")
    month = (match.group("suffix") or "").strip()
    if _MONTH_RE.fullmatch(month):
        normalized_date = _normalize_named_month_date(match, normalizer)
        if normalized_date is not None:
            return normalized_date

    date_candidate = _russian_date_candidate(core)
    if date_candidate is not None:
        candidate = _replace_group(source, match, "core", date_candidate)
        return _finish_normalized(source, normalizer(candidate).strip())

    if "." in core:
        if _is_version(text, match):
            spoken = " точка ".join(
                _normalize_version_part(part, normalizer) for part in core.split(".")
            )
            return _replace_group(source, match, "core", spoken)
        if core.count(".") > 1:
            spoken = " точка ".join(
                _normalize_version_part(part, normalizer) for part in core.split(".")
            )
            return _replace_group(source, match, "core", spoken)
        fractional = core.rsplit(".", 1)[1]
        if not (_CURRENCY_RE.search(source) and len(fractional) == 2):
            candidate = _replace_group(source, match, "core", core.replace(".", ","))
            return _finish_normalized(source, normalizer(candidate).strip())

    return _finish_normalized(source, normalizer(source).strip())


def _normalize_named_month_date(
    match: re.Match[str],
    normalizer: NumberNormalizer,
) -> str | None:
    core = match.group("core")
    if not core.isdecimal() or not 1 <= int(core) <= 31:
        return None
    symbol = (match.group("symbol") or "").strip()
    if symbol:
        return None

    day = int(core)
    context = match.group("context") or ""
    context_word = context.strip().casefold()
    explicit_ordinal = (match.group("ordinal") or "").casefold()
    if explicit_ordinal == "-го" or context_word in _DATE_GENITIVE_CONTEXTS:
        spoken_day = normalizer(f"{day}-го").strip()
    elif explicit_ordinal == "-му" or context_word in _DATE_DATIVE_CONTEXTS:
        spoken_day = normalizer(f"{day}-му").strip()
    elif explicit_ordinal == "-м" or context_word in _DATE_PREPOSITIONAL_CONTEXTS:
        spoken_day = normalizer(f"{day}-м").strip()
    else:
        spoken_day = _ordinal_neuter(day, normalizer)
    month = (match.group("suffix") or "").strip()
    return f"{context}{spoken_day} {month}"


def _ordinal_neuter(value: int, normalizer: NumberNormalizer) -> str:
    genitive = normalizer(f"{value}-го").strip()
    prefix, separator, last_word = genitive.rpartition(" ")
    if last_word.endswith("ого"):
        last_word = last_word[:-3] + "ое"
    elif last_word.endswith("его"):
        last_word = last_word[:-3] + "е"
    return f"{prefix}{separator}{last_word}"


def _russian_date_candidate(core: str) -> str | None:
    parts = re.split(r"[./-]", core)
    if len(parts) != 3 or not all(part.isdecimal() for part in parts):
        return None
    first, second, third = (int(part) for part in parts)
    if len(parts[2]) == 4 and 1 <= first <= 31 and 1 <= second <= 12:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"
    if len(parts[0]) == 4 and 1 <= second <= 12 and 1 <= third <= 31:
        return f"{parts[2]}.{parts[1]}.{parts[0]}"
    return None


def _is_version(text: str, match: re.Match[str]) -> bool:
    core_start, core_end = match.span("core")
    left = text[max(0, core_start - 32) : core_start]
    right = text[core_end : core_end + 2]
    if _VERSION_WORD_RE.search(left):
        return True
    if left and left[-1].isascii() and (left[-1].isalnum() or left[-1] == "_"):
        return True
    if len(left) >= 2 and left[-1] == "-" and left[-2].isascii() and left[-2].isalnum():
        return True
    return bool(right and right[0] == "-" and len(right) > 1 and right[1].isascii())


def _normalize_version_part(part: str, normalizer: NumberNormalizer) -> str:
    if len(part) > 1 and part.startswith("0"):
        return " ".join(normalizer(digit).strip() for digit in part)
    return normalizer(part).strip()


def _replace_group(
    source: str,
    match: re.Match[str],
    group: str,
    replacement: str,
) -> str:
    group_start, group_end = match.span(group)
    relative_start = group_start - match.start()
    relative_end = group_end - match.start()
    return source[:relative_start] + replacement + source[relative_end:]


def _finish_normalized(source: str, value: str) -> str:
    if value.startswith("+"):
        value = "плюс " + value[1:].lstrip()
    elif value.startswith("-"):
        value = "минус " + value[1:].lstrip()
    if source.endswith(".") and not value.endswith((".", "!", "?")):
        value += "."
    return value


def _replace_equals(value: str) -> str:
    protected = tuple(match.span() for match in _PROTECTED_RE.finditer(value))
    output: list[str] = []
    cursor = 0
    for match in _SINGLE_EQUALS_RE.finditer(value):
        if _overlaps(match.span(), protected):
            continue
        output.append(value[cursor : match.start()])
        previous = value[match.start() - 1] if match.start() > 0 else ""
        following = value[match.end()] if match.end() < len(value) else ""
        leading = " " if previous and not previous.isspace() and previous not in "([{«" else ""
        trailing = (
            " "
            if following and not following.isspace() and following not in ".,;:!?)]}»"
            else ""
        )
        output.append(f"{leading}равно{trailing}")
        cursor = match.end()
    if cursor == 0:
        return value
    output.append(value[cursor:])
    return "".join(output)


def _overlaps(span: tuple[int, int], protected: tuple[tuple[int, int], ...]) -> bool:
    start, end = span
    return any(
        start < protected_end and protected_start < end
        for protected_start, protected_end in protected
    )


def create_russian_text_normalizer(*, enabled: bool) -> TextNormalizer | None:
    """Load rutextnorm in Listener and return a fail-open shared normalizer."""

    if not enabled:
        return None
    try:
        from rutextnorm import normalize_russian
    except ImportError as exc:
        raise RuntimeError(
            "Russian TTS text normalization requires rutextnorm in the Listener environment"
        ) from exc

    def normalize(text: str) -> str:
        value = str(text or "")
        try:
            return normalize_russian_numeric_spans(value, normalize_russian)
        except Exception as exc:  # noqa: BLE001 - TTS must fail open on TN errors
            log.warning("Russian TTS text normalization failed: %s", exc)
            return value

    return normalize


__all__ = [
    "NumberNormalizer",
    "TextNormalizer",
    "create_russian_text_normalizer",
    "normalize_russian_numeric_spans",
]
