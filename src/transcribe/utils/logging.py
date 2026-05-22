"""
Logging configuration utilities for the transcription engine.

Provides :class:`ColoredFormatter` for colourised console output and
:func:`setup_logging` for one-shot root logger configuration.

Typical usage::

    from transcribe.utils.logging import setup_logging

    setup_logging(level="INFO")

    logger = logging.getLogger("transcribe")
    logger.info("Engine started")   # → green "INFO" prefix
    logger.warning("Low disk")      # → yellow "WARNING"
    logger.error("Failed")          # → red "ERROR"
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# ANSI colour constants
# ═══════════════════════════════════════════════════════════════

_RESET = "\033[0m"

_BOLD = "\033[1m"
_DIM = "\033[2m"

_BLACK = "\033[30m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"

_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"
_BRIGHT_CYAN = "\033[96m"

# ═══════════════════════════════════════════════════════════════
# Colour scheme per log level
# ═══════════════════════════════════════════════════════════════

_LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: _DIM + _WHITE,
    logging.INFO: _GREEN,
    logging.WARNING: _BOLD + _YELLOW,
    logging.ERROR: _BOLD + _RED,
    logging.CRITICAL: _BOLD + _BRIGHT_RED + _WHITE,  # bright red bg
}

_LEVEL_NAMES: dict[int, str] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


# ═══════════════════════════════════════════════════════════════
# ColoredFormatter
# ═══════════════════════════════════════════════════════════════


class ColoredFormatter(logging.Formatter):
    """Log formatter that applies ANSI colour codes to the level name.

    The output format is::

        [2026-05-22 15:33:00] [32mINFO[0m     my_logger  Message text here  # noqa: W291
        [2026-05-22 15:33:01] [33mWARNING[0m  my_logger  Something fishy

    where the level name (``INFO``, ``WARNING``, etc.) is coloured.

    Parameters
    ----------
    fmt:
        ``logging.Formatter``-style format string.
        Default: ``"[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"``
    datefmt:
        ``time.strftime``-style date format.
        Default: ``"%Y-%m-%d %H:%M:%S"``
    use_colors:
        If ``True`` (default), emit ANSI colour codes.  Set to ``False``
        to produce plain text (useful when writing to files).
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        *,
        use_colors: bool = True,
    ) -> None:
        if fmt is None:
            fmt = "[%(asctime)s] %(levelname)-8s %(name)s  %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"

        super().__init__(fmt=fmt, datefmt=datefmt)
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, applying colour to the level name."""
        if not self._use_colors:
            return super().format(record)

        # Save original values to restore after formatting
        original_levelname = record.levelname
        original_msg = record.msg

        try:
            # Apply colour to the level name
            color = _LEVEL_COLORS.get(record.levelno, _RESET)
            level_label = _LEVEL_NAMES.get(record.levelno, record.levelname)
            record.levelname = f"{color}{level_label}{_RESET}"

            return super().format(record)
        finally:
            # Restore original values (important for reused LogRecords)
            record.levelname = original_levelname
            record.msg = original_msg


# ═══════════════════════════════════════════════════════════════
# setup_logging
# ═══════════════════════════════════════════════════════════════


def setup_logging(
    level: int | str = logging.INFO,
    log_format: str | None = None,
    date_format: str | None = None,
    *,
    use_colors: bool = True,
    stream=None,
) -> None:
    """Configure the root logger with a :class:`ColoredFormatter`.

    This is a convenience function that:

    - Creates a :class:`ColoredFormatter` with the given format / date format.
    - Attaches a ``StreamHandler`` to the root logger.
    - Sets the root logger level.

    It is **idempotent for the same level** — if the root logger already
    has handlers configured (e.g. imported in multiple places), it will
    **remove any existing handlers** and replace them, guaranteeing
    consistent formatting.

    Parameters
    ----------
    level:
        Log level (``logging.DEBUG``, ``logging.INFO``, etc. or a
        case-insensitive string like ``"debug"``, ``"INFO"``).
        Default ``logging.INFO``.
    log_format:
        ``logging.Formatter``-style format string.  If ``None``, uses
        the default from :class:`ColoredFormatter`.
    date_format:
        ``time.strftime``-style date format.  If ``None``, uses the
        default from :class:`ColoredFormatter` (``"%Y-%m-%d %H:%M:%S"``).
    use_colors:
        Passed through to :class:`ColoredFormatter`.  Default ``True``.
    stream:
        Output stream (defaults to ``sys.stderr``).

    Example
    -------
    ::

        setup_logging("DEBUG")
        logger = logging.getLogger("transcribe")
        logger.debug("Tracing enabled")
    """
    if isinstance(level, str):
        level = _resolve_level(level)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicate output
    root.handlers.clear()

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(level)

    formatter = ColoredFormatter(
        fmt=log_format,
        datefmt=date_format,
        use_colors=use_colors,
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def _resolve_level(name: str) -> int:
    """Convert a level string (``\"debug\"``, ``\"INFO\"``, etc.) to
    its ``logging`` constant.

    Raises ``ValueError`` if the name is not recognised.
    """
    normalized = name.upper().strip()
    level = getattr(logging, normalized, None)
    if not isinstance(level, int):
        raise ValueError(
            f"Unknown log level: {name!r}. "
            f"Expected one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
        )
    return level
