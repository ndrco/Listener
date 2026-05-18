from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Tuple

try:  # pragma: no cover - import availability depends on environment
    import torch
except Exception:  # pragma: no cover - torch is optional for non-ML code paths
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - import availability depends on environment
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - import availability depends on environment
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

from core.config import cfg
from core import perf

log = logging.getLogger(__name__)
_IDENTITY_NAME_RE = re.compile(r"^\s*(Name|Имя)\s*[:=\-—]\s*(.+?)\s*$", re.IGNORECASE)
_LOCAL_MUTE_COMMANDS = frozenset({"замолчи", "молчи", "тихо", "помолчи"})
_LOCAL_NORMAL_COMMANDS = frozenset(
    {"голос", "говори", "разговаривай", "ответь", "ты где", "слушай"}
)
_LOCAL_STANDBY_COMMANDS = frozenset(
    {
        "отключись",
        "выключись",
        "перестань слушать",
        "перестань разговаривать",
        "не слушай",
    }
)
_LOCAL_ABORT_COMMANDS = frozenset({"остановись", "стоп", "хватит", "прекрати", "достаточно", "стой"})
_LOCAL_BARGE_IN_COMMANDS = frozenset(
    {
        "нет",
        "не так",
        "подожди",
        "постой",
        "секунду",
        "я имел в виду",
        "точнее",
        "не это",
        "другими словами",
    }
)
_LOCAL_COMMAND_FILLERS = frozenset(
    {
        "пожалуйста",
        "плиз",
        "ладно",
        "сейчас",
        "снова",
        "обратно",
        "давай",
        "ну",
    }
)


class SpeechGateMode(str, Enum):
    STANDBY = "standby"
    MUTE = "mute"
    NORMAL = "normal"
    CHATTY = "chatty"

    @classmethod
    def from_value(cls, value: object) -> "SpeechGateMode":
        text = str(value or "").strip().lower()
        for mode in cls:
            if mode.value == text:
                return mode
        return cls.NORMAL

    @classmethod
    def parse(cls, value: object) -> "SpeechGateMode":
        text = str(value or "").strip().lower()
        for mode in cls:
            if mode.value == text:
                return mode
        valid = ", ".join(mode.value for mode in cls)
        raise ValueError(f"invalid speech gate mode {value!r}; expected one of: {valid}")


@dataclass
class GateDecision:
    allowed: bool
    rule_score: float
    ml_score: float
    final_score: float
    continuation: bool
    reason: str


@dataclass(frozen=True)
class LocalCommandMatch:
    action: str
    assistant_name: str
    phrase: str


class DirectedIntentClassifier:
    """Wrapper around an Electra directed-speech classifier."""

    def __init__(self, model_path: str, device: str = "cpu", *, max_length: int = 64) -> None:
        if torch is None:
            raise RuntimeError("torch is not available")
        if AutoTokenizer is None or AutoModelForSequenceClassification is None:
            raise RuntimeError("transformers is not available")
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    def predict_directed_prob(self, text: str) -> float:
        encoded = self.tokenizer(
            [text],
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[1].item()) if probs.numel() >= 2 else 0.0


class SpeechDirectionGate:
    """Two-stage gate for directed user phrases."""

    def __init__(self) -> None:
        sg_cfg = getattr(cfg, "speech_gate", object())
        self.root_path = Path(getattr(getattr(cfg, "paths", object()), "root", Path(".")))

        self.enabled = bool(getattr(sg_cfg, "enable", True))
        self.mode = SpeechGateMode.from_value(getattr(sg_cfg, "mode", SpeechGateMode.NORMAL.value))
        self.rules_threshold = float(getattr(sg_cfg, "rules_threshold", 0.7))
        self.final_threshold = float(getattr(sg_cfg, "final_threshold", 0.5))
        self.attention_window = float(getattr(sg_cfg, "attention_window_seconds", 8.0))
        self.attention_extension = float(getattr(sg_cfg, "attention_extension_seconds", 3.0))

        self.patterns = self._load_patterns(sg_cfg)
        self._classifier: DirectedIntentClassifier | None = None
        self._classifier_disabled = False
        self._classifier_failed_logged = False
        self._ml_threshold = float(getattr(getattr(sg_cfg, "model", object()), "threshold", 0.7))
        self._classifier_device = "cpu"
        self._classifier_path: Path | None = None
        self._classifier_max_length = 64

        self._attention_until: float = 0.0

    @classmethod
    def from_config(cls) -> "SpeechDirectionGate":
        return cls()

    def warmup(self) -> bool:
        """Load the classifier and run one tiny inference before speech arrives."""

        if not self.enabled or self.mode == SpeechGateMode.CHATTY:
            return False

        model_cfg = getattr(getattr(cfg, "speech_gate", object()), "model", object())
        model_path = Path(str(getattr(model_cfg, "path", "models/directed-ruElectra-small-fp16")))
        if not model_path.is_absolute():
            model_path = self.root_path / model_path
        if not model_path.exists():
            return False

        start_ns = perf.now_ns()
        classifier = self._load_classifier()
        classifier.predict_directed_prob("привет")
        perf.emit(
            "speech_gate",
            "warmup",
            duration_ms=perf.elapsed_ms(start_ns),
        )
        return True

    def set_mode(self, mode: SpeechGateMode) -> None:
        self.mode = mode

    def clear_attention(self) -> None:
        self._attention_until = 0.0

    def reload_patterns(self) -> None:
        sg_cfg = getattr(cfg, "speech_gate", object())
        self.patterns = self._load_patterns(sg_cfg)

    def _load_patterns(self, sg_cfg) -> dict[str, set[str]]:
        def _as_set(values: Iterable[str] | None) -> set[str]:
            return {str(v).strip().lower() for v in values or [] if str(v).strip()}

        file_groups = self._load_patterns_from_file(getattr(sg_cfg, "patterns_file", None))
        identity_names = self._load_identity_names(getattr(sg_cfg, "identity_file", None))
        groups = {
            "assistant_names": _as_set(getattr(sg_cfg, "assistant_names", None)),
            "command_verbs": _as_set(getattr(sg_cfg, "command_verbs", None)),
            "politeness_markers": _as_set(getattr(sg_cfg, "politeness_markers", None)),
            "question_markers": _as_set(getattr(sg_cfg, "question_markers", None)),
            "modal_markers": _as_set(getattr(sg_cfg, "modal_markers", None)),
            "continuation_patterns": _as_set(getattr(sg_cfg, "continuation_patterns", None)),
        }
        for kind, values in file_groups.items():
            groups.setdefault(kind, set()).update(values)
        groups.setdefault("assistant_names", set()).update(identity_names)
        return {kind: values for kind, values in groups.items() if values}

    def _load_patterns_from_file(self, path_str: str | None) -> dict[str, set[str]]:
        if not path_str:
            return {}

        path = Path(path_str)
        if not path.is_absolute():
            path = self.root_path / path

        try:
            if not path.exists():
                log.warning("speech_gate: patterns file not found: %s", path)
                return {}
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                log.warning("speech_gate: patterns file has invalid format (dict expected): %s", path)
                return {}
            cleaned: dict[str, set[str]] = {}
            for key, values in data.items():
                if key == "assistant_names":
                    log.warning(
                        "speech_gate: assistant_names in %s are ignored; use %s instead",
                        path,
                        self._identity_path_display(),
                    )
                    continue
                if not isinstance(values, list):
                    continue
                normalized = {str(item).strip().lower() for item in values if str(item).strip()}
                if normalized:
                    cleaned[str(key)] = normalized
            return cleaned
        except Exception as exc:
            log.warning("speech_gate: failed to read patterns file %s: %s", path, exc)
            return {}

    def _load_identity_names(self, path_str: str | None) -> set[str]:
        path = self._resolve_identity_path(path_str)
        if path is None:
            return set()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            log.info("speech_gate: identity file not found: %s", path)
            return set()
        except OSError as exc:
            log.warning("speech_gate: failed to read identity file %s: %s", path, exc)
            return set()

        names: set[str] = set()
        for raw_line in content.splitlines():
            names.update(self._parse_identity_line(raw_line))
        if names:
            log.info("speech_gate: assistant names loaded from %s (%d)", path, len(names))
        return names

    def _resolve_identity_path(self, path_str: str | None) -> Path | None:
        if path_str and str(path_str).strip().lower() != "auto":
            path = Path(str(path_str)).expanduser()
            if not path.is_absolute():
                path = self.root_path / path
            return path

        for candidate in self._discover_openclaw_identity_paths():
            if candidate.is_file():
                return candidate
        log.info("speech_gate: OpenClaw identity file not found; set speech_gate.identity_file")
        return None

    def _discover_openclaw_identity_paths(self) -> list[Path]:
        candidates: list[Path] = []

        direct = os.environ.get("OPENCLAW_IDENTITY_FILE")
        if direct:
            candidates.append(Path(direct).expanduser())

        workspace = os.environ.get("OPENCLAW_WORKSPACE")
        if workspace:
            candidates.append(Path(workspace).expanduser() / "IDENTITY.md")

        state_dir = os.environ.get("OPENCLAW_STATE_DIR")
        if state_dir:
            candidates.append(Path(state_dir).expanduser() / "workspace" / "IDENTITY.md")

        config_paths: list[Path] = []
        config_env = os.environ.get("OPENCLAW_CONFIG_PATH")
        if config_env:
            config_paths.append(Path(config_env).expanduser())
        home = Path.home()
        config_paths.append(home / ".openclaw" / "openclaw.json")
        config_paths.append(home / ".openclaw-dev" / "openclaw.json")
        config_paths.extend(sorted(home.glob(".openclaw-*/openclaw.json")))

        for config_path in config_paths:
            workspace_path = self._read_openclaw_workspace_from_config(config_path)
            if workspace_path is not None:
                candidates.append(workspace_path / "IDENTITY.md")

        candidates.append(home / ".openclaw" / "workspace" / "IDENTITY.md")
        candidates.append(home / ".openclaw-dev" / "workspace" / "IDENTITY.md")
        candidates.extend(sorted(home.glob(".openclaw-*/workspace/IDENTITY.md")))

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                resolved = candidate.expanduser()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(resolved)
        return unique

    @staticmethod
    def _read_openclaw_workspace_from_config(config_path: Path) -> Path | None:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        try:
            workspace = data["agents"]["defaults"]["workspace"]
        except (KeyError, TypeError):
            return None
        if not isinstance(workspace, str) or not workspace.strip():
            return None
        return Path(workspace.strip()).expanduser()

    @classmethod
    def _parse_identity_line(cls, line: str) -> set[str]:
        text = line.strip()
        text = re.sub(r"^\s*[-*]\s*", "", text)
        text = re.sub(r"^\s*#{1,6}\s*", "", text)
        text = text.replace("**", "")
        match = _IDENTITY_NAME_RE.match(text)
        if not match:
            return set()
        return cls._split_identity_names(match.group(2))

    @staticmethod
    def _split_identity_names(value: str) -> set[str]:
        text = value.strip().strip("`*_\"'«»“”")
        if not text:
            return set()
        result: set[str] = set()
        for part in re.split(r"[,;/|]", text):
            cleaned = part.strip().strip("`*_\"'«»“”")
            if cleaned:
                result.add(cleaned.lower())
        return result

    def _identity_path_display(self) -> str:
        return "speech_gate.identity_file or OpenClaw config workspace/IDENTITY.md"

    def _load_classifier(self) -> DirectedIntentClassifier:
        if self._classifier is not None:
            return self._classifier
        if self._classifier_disabled:
            raise RuntimeError("classifier disabled")

        model_cfg = getattr(getattr(cfg, "speech_gate", object()), "model", object())
        model_path = Path(str(getattr(model_cfg, "path", "models/directed-ruElectra-small-fp16")))
        if not model_path.is_absolute():
            model_path = self.root_path / model_path
        if not model_path.exists():
            self._classifier_disabled = True
            raise FileNotFoundError(f"speech gate model path does not exist: {model_path}")

        device = str(getattr(model_cfg, "device", "cpu"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            log.warning("speech_gate: cuda requested but unavailable, falling back to cpu")
            device = "cpu"

        max_len = int(getattr(model_cfg, "max_length", 64))
        self._classifier_path = model_path
        self._classifier_max_length = max(1, max_len)
        self._classifier = self._create_classifier_with_fallback(device=device)
        return self._classifier

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "cuda out of memory",
                "out of memory",
                "failed to allocate memory",
            )
        )

    def _create_classifier(self, *, device: str) -> DirectedIntentClassifier:
        if self._classifier_path is None:
            raise RuntimeError("classifier path is not initialised")
        classifier = DirectedIntentClassifier(
            str(self._classifier_path),
            device=device,
            max_length=self._classifier_max_length,
        )
        self._classifier_device = device
        return classifier

    def _create_classifier_with_fallback(self, *, device: str) -> DirectedIntentClassifier:
        try:
            return self._create_classifier(device=device)
        except Exception as exc:
            if not device.startswith("cuda") or not self._is_cuda_oom(exc):
                raise
            log.warning(
                "speech_gate: classifier CUDA initialisation ran out of memory; retrying on cpu"
            )
            return self._create_classifier(device="cpu")

    def _retry_classifier_on_cpu(self) -> DirectedIntentClassifier:
        self._classifier = self._create_classifier_with_fallback(device="cpu")
        return self._classifier

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = (text or "").strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def _contains(text: str, patterns: Iterable[str]) -> bool:
        for pattern in patterns:
            if not pattern:
                continue
            if re.search(rf"\b{re.escape(pattern)}\b", text):
                return True
            if pattern in text:
                return True
        return False

    def _rules_score(self, text: str) -> Tuple[float, bool]:
        score = 0.0
        has_name = self._contains(text, self.patterns.get("assistant_names", set()))
        if has_name:
            return 1.0, True

        if self._contains(text, self.patterns.get("command_verbs", set())):
            score += 0.3
        if self._contains(text, self.patterns.get("politeness_markers", set())):
            score += 0.1
        if self._contains(text, self.patterns.get("question_markers", set())):
            score += 0.2
        if self._contains(text, self.patterns.get("modal_markers", set())):
            score += 0.2
        if "?" in text:
            score += 0.1

        return min(score, 1.0), has_name

    def _detect_continuation(self, text: str) -> bool:
        stripped = text.rstrip()
        for pattern in self.patterns.get("continuation_patterns", set()):
            if not pattern:
                continue
            if stripped.endswith(pattern):
                return True
            if re.search(rf"{re.escape(pattern)}\s*$", stripped):
                return True
        return False

    @staticmethod
    def _strip_command_tail(text: str) -> str:
        return text.strip(" \t\r\n,.;:!?-—()[]{}\"'")

    def _strip_optional_local_fillers(self, text: str) -> str:
        current = self._strip_command_tail(text)
        if not current:
            return ""

        fillers = (
            self.patterns.get("politeness_markers", set()) | _LOCAL_COMMAND_FILLERS
        )
        while current:
            matched = False
            for filler in sorted(fillers, key=len, reverse=True):
                if not filler:
                    continue
                match = re.match(rf"^{re.escape(filler)}(?:\b|$)", current)
                if match is None:
                    continue
                current = self._strip_command_tail(current[match.end() :])
                matched = True
                break
            if not matched:
                break
        return current

    def find_leading_assistant_name(self, text: str) -> str | None:
        normalized = self._normalize(text)
        if not normalized:
            return None
        assistant_names = sorted(
            self.patterns.get("assistant_names", set()),
            key=len,
            reverse=True,
        )
        for name in assistant_names:
            if not name:
                continue
            if re.match(rf"^{re.escape(name)}(?:\b|$)", normalized):
                return name
        return None

    def _find_command_tail_after_leading_name(self, normalized_text: str) -> tuple[str, str] | None:
        assistant_names = sorted(
            self.patterns.get("assistant_names", set()),
            key=len,
            reverse=True,
        )
        for name in assistant_names:
            if not name:
                continue
            match = re.match(rf"^{re.escape(name)}(?:\b|$)", normalized_text)
            if match is None:
                continue
            tail = self._strip_command_tail(normalized_text[match.end() :])
            if tail:
                return name, tail
        return None

    def _match_local_command_phrase(self, tail: str, phrases: Iterable[str]) -> str | None:
        for phrase in sorted(set(phrases), key=len, reverse=True):
            if not phrase:
                continue
            match = re.match(rf"^{re.escape(phrase)}(?:\b|$)", tail)
            if match is None:
                continue
            rest = self._strip_optional_local_fillers(tail[match.end() :])
            if not rest:
                return phrase
        return None

    def _match_local_command_prefix(self, tail: str, phrases: Iterable[str]) -> str | None:
        for phrase in sorted(set(phrases), key=len, reverse=True):
            if not phrase:
                continue
            if re.match(rf"^{re.escape(phrase)}(?:\b|$)", tail):
                return phrase
        return None

    def detect_local_command(self, text: str) -> LocalCommandMatch | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        name_and_tail = self._find_command_tail_after_leading_name(normalized)
        if name_and_tail is None:
            return None
        assistant_name, tail = name_and_tail
        command_tail = self._strip_optional_local_fillers(tail)
        if not command_tail:
            return None

        checks = (
            ("mute", self.patterns.get("local_mute_commands", set()) | _LOCAL_MUTE_COMMANDS),
            (
                "standby",
                self.patterns.get("local_standby_commands", set()) | _LOCAL_STANDBY_COMMANDS,
            ),
            (
                "stop_generation",
                self.patterns.get("local_abort_commands", set()) | _LOCAL_ABORT_COMMANDS,
            ),
            ("normal", self.patterns.get("local_normal_commands", set()) | _LOCAL_NORMAL_COMMANDS),
        )
        for action, phrases in checks:
            phrase = self._match_local_command_phrase(command_tail, phrases)
            if phrase:
                return LocalCommandMatch(
                    action=action,
                    assistant_name=assistant_name,
                    phrase=phrase,
                )
        return None

    def detect_barge_in_command(self, text: str) -> LocalCommandMatch | None:
        normalized = self._normalize(text)
        if not normalized:
            return None

        name_and_tail = self._find_command_tail_after_leading_name(normalized)
        if name_and_tail is None:
            return None
        assistant_name, tail = name_and_tail
        command_tail = self._strip_optional_local_fillers(tail)
        if not command_tail:
            return None

        phrase = self._match_local_command_prefix(
            command_tail,
            self.patterns.get("local_barge_in_commands", set()) | _LOCAL_BARGE_IN_COMMANDS,
        )
        if not phrase:
            return None
        return LocalCommandMatch(
            action="barge_in",
            assistant_name=assistant_name,
            phrase=phrase,
        )

    def _extend_attention(self, now: float) -> None:
        self._attention_until = max(self._attention_until, now) + self.attention_extension

    def _activate_attention(self, now: float, continuation: bool) -> None:
        self._attention_until = now + self.attention_window
        if continuation:
            self._extend_attention(now)

    def should_allow(
        self,
        text: str,
        *,
        payload: dict | None = None,
        mode: SpeechGateMode | None = None,
    ) -> GateDecision:
        del payload  # reserved for future context-aware gating

        normalized = self._normalize(text)
        continuation = self._detect_continuation(normalized)
        now = time.monotonic()
        active_mode = mode or self.mode

        if not self.enabled:
            return GateDecision(True, 0.0, 0.0, 1.0, continuation, "disabled")

        if not normalized:
            return GateDecision(False, 0.0, 0.0, 0.0, continuation, "empty")

        if active_mode == SpeechGateMode.STANDBY:
            return GateDecision(False, 0.0, 0.0, 0.0, continuation, "standby")

        rule_score, has_name = self._rules_score(normalized)
        has_leading_name = self.find_leading_assistant_name(normalized) is not None

        if active_mode == SpeechGateMode.MUTE and not has_leading_name:
            return GateDecision(False, rule_score, 0.0, rule_score, continuation, "mute")

        in_attention = active_mode == SpeechGateMode.CHATTY or (
            active_mode == SpeechGateMode.NORMAL and now < self._attention_until
        )

        if in_attention:
            if continuation:
                self._extend_attention(now)
            return GateDecision(True, rule_score, 1.0, 1.0, continuation, "attention")

        if rule_score >= self.rules_threshold:
            final_score = max(rule_score, self.final_threshold)
            self._activate_attention(now, continuation)
            return GateDecision(True, rule_score, 1.0, final_score, continuation, "rules")

        try:
            classifier = self._load_classifier()
            try:
                ml_score = classifier.predict_directed_prob(normalized)
            except Exception as exc:
                if not self._classifier_device.startswith("cuda") or not self._is_cuda_oom(exc):
                    raise
                log.warning(
                    "speech_gate: classifier CUDA inference ran out of memory; retrying on cpu"
                )
                classifier = self._retry_classifier_on_cpu()
                ml_score = classifier.predict_directed_prob(normalized)
        except Exception as exc:
            if not self._classifier_failed_logged:
                log.warning("speech_gate: classifier unavailable, using rules-only fallback: %s", exc)
                self._classifier_failed_logged = True
            final_score = rule_score
            allowed = final_score >= self.final_threshold
            if allowed:
                self._activate_attention(now, continuation)
                return GateDecision(
                    True,
                    rule_score,
                    0.0,
                    final_score,
                    continuation,
                    "rules_fallback",
                )
            return GateDecision(False, rule_score, 0.0, final_score, continuation, "low_score")

        ml_pass = ml_score >= self._ml_threshold
        final_score = 0.6 * ml_score + 0.4 * rule_score
        allowed = ml_pass and final_score >= self.final_threshold
        if allowed:
            self._activate_attention(now, continuation)
            return GateDecision(True, rule_score, ml_score, final_score, continuation, "ml")

        return GateDecision(False, rule_score, ml_score, final_score, continuation, "low_score")


__all__ = [
    "DirectedIntentClassifier",
    "GateDecision",
    "LocalCommandMatch",
    "SpeechDirectionGate",
    "SpeechGateMode",
]
