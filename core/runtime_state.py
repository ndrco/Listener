"""Helpers for persisting small runtime control state across restarts."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from core.config import cfg

log = logging.getLogger(__name__)


class RuntimeStateStore:
    """Small JSON store for runtime control state."""

    _WRITE_LOCK = threading.Lock()

    def __init__(self, path: str | Path | None) -> None:
        if path in (None, ""):
            self.path: Path | None = None
            return
        path_obj = Path(path).expanduser()
        if not path_obj.is_absolute():
            path_obj = cfg.paths.root / path_obj
        self.path = path_obj

    @classmethod
    def from_config(cls) -> "RuntimeStateStore":
        control_cfg = getattr(cfg, "control", object())
        return cls(getattr(control_cfg, "state_path", None))

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def load(self) -> dict[str, Any]:
        path = self.path
        if path is None or not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - broken state must not stop startup
            log.warning("runtime_state: failed to read %s: %s", path, exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def get_section(self, name: str) -> dict[str, Any] | None:
        payload = self.load()
        section = payload.get(str(name))
        return section if isinstance(section, dict) else None

    def save_section(self, name: str, section: dict[str, Any] | None) -> None:
        path = self.path
        if path is None:
            return
        with self._WRITE_LOCK:
            payload = self.load()
            payload["version"] = 1
            payload["saved_at"] = time.time()
            section_name = str(name)
            if section is None:
                payload.pop(section_name, None)
            else:
                payload[section_name] = dict(section)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(path)


__all__ = ["RuntimeStateStore"]
