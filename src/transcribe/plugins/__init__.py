"""
transcribe.plugins — plugin system for the transcription engine.

Provides the base plugin class (:class:`AbstractPlugin`) and a discovery
mechanism (:class:`PluginManager`) that loads plugins registered via the
``transcribe.plugins`` entry-point group in ``pyproject.toml``.
"""

from __future__ import annotations

from .base import AbstractPlugin
from .plugin_manager import PluginManager

__all__ = [
    "AbstractPlugin",
    "PluginManager",
]
