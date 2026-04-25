from __future__ import annotations

import logging
import sys
from typing import Optional

_YELLOW = "\x1b[33m"
_RESET = "\x1b[0m"

_PROJECT_LOGGERS = (
    "agents",
    "audio",
    "core",
    "utils",
)
_PROJECT_DEBUG_HANDLER_NAME = "project_debug"


class _OnlyDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.DEBUG


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: Optional[str], *, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self._use_color:
            return message
        if record.levelno >= logging.WARNING:
            return f"{_YELLOW}{message}{_RESET}"
        return message


def configure_logging(*, debug: bool, info: bool) -> None:
    base_level = logging.WARNING
    if debug or info:
        base_level = logging.INFO

    root = logging.getLogger()
    stream = sys.stderr
    use_color = stream.isatty()
    formatter = ColorFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt=None,
        use_color=use_color,
    )

    if root.handlers:
        root.setLevel(base_level)
        for handler in root.handlers:
            handler.setLevel(base_level)
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(stream)
        handler.setLevel(base_level)
        handler.setFormatter(formatter)
        root.setLevel(base_level)
        root.addHandler(handler)

    project_level = logging.DEBUG if debug else logging.INFO
    debug_handler = None
    if debug:
        debug_handler = logging.StreamHandler(stream)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        debug_handler.addFilter(_OnlyDebugFilter())
        debug_handler.name = _PROJECT_DEBUG_HANDLER_NAME
    for name in _PROJECT_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(project_level)
        if debug_handler is not None:
            if not any(
                handler.name == _PROJECT_DEBUG_HANDLER_NAME
                for handler in logger.handlers
            ):
                logger.addHandler(debug_handler)
