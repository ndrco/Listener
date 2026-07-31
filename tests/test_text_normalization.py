import logging

import speaker.text_normalization as text_normalization
from speaker.text_normalization import (
    create_russian_text_normalizer,
    normalize_russian_numeric_spans,
)


def test_russian_number_normalization_preserves_non_numeric_text():
    replacements = {
        "В 2026 году": "В две тысячи двадцать шестом году",
        "3": "три",
        "5": "пять",
        "1 500 рублей": "тысяча пятьсот рублей",
        "на 87%": "на восемьдесят семь процентов",
        "6": "шесть",
    }

    result = normalize_russian_numeric_spans(
        "В 2026 году версия 3.5 стоит 1 500 рублей и готова на 87%. "
        "GPT-5.6-terra остаётся OpenClaw.",
        replacements.__getitem__,
    )

    assert result == (
        "В две тысячи двадцать шестом году версия три точка пять стоит "
        "тысяча пятьсот рублей и готова на восемьдесят семь процентов. "
        "GPT-пять точка шесть-terra остаётся OpenClaw."
    )


def test_russian_number_normalization_skips_text_without_digits():
    def unexpected_call(_value: str) -> str:
        raise AssertionError("normalizer must not be called")

    text = "OpenClaw и Google остаются без транслитерации."

    assert normalize_russian_numeric_spans(text, unexpected_call) == text


def test_russian_number_normalization_protects_machine_readable_spans():
    def unexpected_call(_value: str) -> str:
        raise AssertionError("protected numeric spans must not be normalized")

    text = (
        "https://example.com/v3.5?id=2026, IP 192.168.1.1, "
        "+7 999 123-45-67 и `/opt/model-3.5/file`."
    )

    assert normalize_russian_numeric_spans(text, unexpected_call) == text


def test_russian_number_normalization_uses_russian_decimal_separator_for_units():
    calls: list[str] = []

    def fake_normalizer(value: str) -> str:
        calls.append(value)
        return "три целых и пять десятых килограмма"

    result = normalize_russian_numeric_spans("Вес — 3.5 кг.", fake_normalizer)

    assert result == "Вес — три целых и пять десятых килограмма."
    assert calls == ["3,5 кг"]


def test_russian_number_normalization_speaks_single_equals_outside_code():
    replacements = {"2": "два", "4": "четыре", "5": "пять"}

    assert normalize_russian_numeric_spans(
        "2+2=4, x=5, x == 5, `x=5`.",
        replacements.__getitem__,
    ) == "два+два равно четыре, x равно пять, x == пять, `x=5`."


def test_russian_number_normalization_supports_named_month_dates():
    replacements = {
        "1-го": "первого",
        "21-го": "двадцать первого",
        "2-му": "второму",
    }

    result = normalize_russian_numeric_spans(
        "1 августа, с 21 мая и к 2 августа.",
        replacements.__getitem__,
    )

    assert result == "первое августа, с двадцать первого мая и к второму августа."


def test_russian_number_normalization_leaves_noun_agreement_to_source_text():
    assert normalize_russian_numeric_spans(
        "2 чашка кофе и 21 сообщение.",
        lambda value: {"2": "два", "21": "двадцать один"}[value],
    ) == "два чашка кофе и двадцать один сообщение."


def test_shared_normalizer_fails_open(monkeypatch, caplog):
    normalizer = create_russian_text_normalizer(enabled=True)
    assert normalizer is not None
    monkeypatch.setattr(
        text_normalization,
        "normalize_russian_numeric_spans",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("bad TN")),
    )

    with caplog.at_level(logging.WARNING):
        result = normalizer("В 2026 году.")

    assert result == "В 2026 году."
    assert "bad TN" in caplog.text
