"""Observability: structured logging built on the standard library.

The system emits two kinds of records:

* **startup events**: one wide event per process describing how it was
  configured (device, model, host, ...);
* **usage events**: one wide event per meaningful operation (an index run, a
  projection, a search, a prediction), carrying *all* of that operation's
  context in a single line rather than scattering it across many log calls.

A "wide event" is a single log record with many structured fields attached, so
one line answers "what happened, to what, how big, how long". Fields are
rendered as ``key=value`` pairs and also attached to the record's ``extra`` so
a future JSON handler could serialize them verbatim.

Everything here uses only ``logging`` from the stdlib, no third-party logging
dependency.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_ROOT_LOGGER_NAME = "cbir"
_CONFIGURED = False

# ANSI colours for the console. Kept minimal and disabled automatically when
# stdout is not a TTY (e.g. piped to a file or running under Docker logs).
_LEVEL_COLORS = {
    logging.DEBUG: "\033[38;5;244m",  # grey
    logging.INFO: "\033[38;5;39m",  # blue
    logging.WARNING: "\033[38;5;214m",  # amber
    logging.ERROR: "\033[38;5;196m",  # red
    logging.CRITICAL: "\033[48;5;196m\033[97m",  # white on red
}
_DIM = "\033[38;5;244m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

# Standard LogRecord attributes we must not treat as wide-event fields.
_RESERVED = set(
    vars(logging.makeLogRecord({})).keys()
) | {"message", "asctime", "taskName"}


class WideEventFormatter(logging.Formatter):
    """Render ``time level logger | message | key=value ...`` with colour."""

    def __init__(self, *, use_color: bool) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        logger = record.name.removeprefix(f"{_ROOT_LOGGER_NAME}.")
        message = record.getMessage()

        fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        field_text = " ".join(f"{key}={_render(value)}" for key, value in fields.items())

        if self.use_color:
            color = _LEVEL_COLORS.get(record.levelno, "")
            head = f"{_DIM}{timestamp}{_RESET} {color}{level:<8}{_RESET} {_DIM}{logger}{_RESET}"
            body = f"{_BOLD}{message}{_RESET}"
            tail = f" {_DIM}{field_text}{_RESET}" if field_text else ""
        else:
            head = f"{timestamp} {level:<8} {logger}"
            body = message
            tail = f" {field_text}" if field_text else ""

        line = f"{head} | {body}{tail}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


def _render(value: Any) -> str:
    """Render a field value compactly (quote strings with spaces)."""
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return f'"{text}"' if " " in text else text


def configure_logging(level: int | str = logging.INFO, *, color: bool | None = None) -> None:
    """Install the wide-event handler on the ``cbir`` logger once.

    Idempotent: calling it again only adjusts the level. ``color`` defaults to
    auto-detection based on whether stdout is a TTY.
    """
    global _CONFIGURED
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)

    if _CONFIGURED:
        return

    use_color = sys.stdout.isatty() if color is None else color
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(WideEventFormatter(use_color=use_color))
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``cbir`` namespace."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def log_startup(component: str, **fields: Any) -> None:
    """Emit a single wide event describing how a component started up."""
    configure_logging()
    get_logger(component).info("startup", extra={"event": f"{component}.startup", **fields})


def log_event(
    logger: logging.Logger, event: str, message: str | None = None, **fields: Any
) -> None:
    """Emit a single wide usage event with arbitrary structured fields."""
    logger.info(message or event, extra={"event": event, **fields})


@contextmanager
def timed_event(logger: logging.Logger, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a block and emit one wide event when it finishes.

    Yields a mutable dict; fields added to it inside the block are merged into
    the final event (e.g. a count only known after the work is done)::

        with timed_event(log, "index.run", collection=name) as ev:
            ev["inserted"] = do_work()
    """
    configure_logging()
    started = time.perf_counter()
    extra: dict[str, Any] = dict(fields)
    try:
        yield extra
    finally:
        extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
        log_event(logger, event, **extra)
