"""
Plugin discovery and lifecycle manager.

Uses Python ``importlib.metadata`` entry points (``transcribe.plugins``
group) so third-party packages can register plugins declaratively in their
``pyproject.toml``::

    [project.entry-points."transcribe.plugins"]
    my_plugin = "my_package.plugin:MyPlugin"
"""

from __future__ import annotations

import logging
import sys
from typing import Sequence

from ..core.interfaces import TranscriptionPlugin
from ..core.models import EngineConfig
from .base import AbstractPlugin

if sys.version_info >= (3, 12):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points  # type: ignore[no-redef]

logger = logging.getLogger("transcribe.plugins")

# ── Entry-point group constant ─────────────────────────────────────
ENTRY_POINT_GROUP = "transcribe.plugins"
"""The ``importlib.metadata`` entry-point group used for plugin discovery."""


class PluginManager:
    """
    Discovers, loads, and manages the lifecycle of transcription plugins.

    Typical usage::

        manager = PluginManager()
        manager.discover()                     # scan entry points
        await manager.setup_all(config)         # call setup() on each
        ...
        audio = await manager.run_pre_process(audio)
        result = await manager.run_post_process(result)
        ...
        await manager.teardown_all()            # cleanup

    Plugins are returned in **priority order** (ascending).  Disabled
    plugins (``plugin.enabled is False``) are still tracked but skipped
    during hook execution.
    """

    def __init__(self) -> None:
        self._plugins: list[AbstractPlugin] = []

    # ── Discovery ──────────────────────────────────────────────────

    def discover(self, group: str = ENTRY_POINT_GROUP) -> list[AbstractPlugin]:
        """
        Scan the ``transcribe.plugins`` entry-point group and instantiate
        all registered plugins.

        Args:
            group: Entry-point group name (default: ``transcribe.plugins``).

        Returns:
            Sorted list of discovered plugins (by priority, ascending).

        Raises:
            TypeError: If an entry point does not resolve to an
                       ``AbstractPlugin`` subclass.
        """
        discovered: list[AbstractPlugin] = []
        eps = entry_points(group=group)

        for ep in sorted(eps, key=lambda e: e.name):
            try:
                plugin_cls = ep.load()
            except Exception as exc:
                logger.error(
                    "Failed to load entry point %s=%s: %s",
                    ep.name,
                    ep.value,
                    exc,
                )
                continue

            if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, AbstractPlugin)):
                logger.error(
                    "Entry point %s=%s does not resolve to an AbstractPlugin subclass "
                    "(got %s). Skipping.",
                    ep.name,
                    ep.value,
                    type(plugin_cls).__name__,
                )
                continue

            try:
                instance = plugin_cls()  # type: ignore[call-arg]
            except Exception as exc:
                logger.error(
                    "Failed to instantiate plugin %s (%s): %s",
                    ep.name,
                    ep.value,
                    exc,
                )
                continue

            discovered.append(instance)
            logger.info(
                "Discovered plugin %s (priority=%d, enabled=%s)",
                instance.name,
                instance.priority,
                instance.enabled,
            )

        self._plugins = self._sort(discovered)
        return self.plugins

    def register(self, plugin: AbstractPlugin) -> None:
        """
        Manually register a plugin instance (bypasses entry-point discovery).

        Args:
            plugin: An ``AbstractPlugin`` instance.
        """
        if not isinstance(plugin, AbstractPlugin):
            raise TypeError(
                f"Expected AbstractPlugin instance, got {type(plugin).__name__}"
            )
        # Prevent duplicates by name
        names = {p.name for p in self._plugins}
        if plugin.name in names:
            logger.warning(
                "Plugin %r already registered — replacing.", plugin.name
            )
            self._plugins = [p for p in self._plugins if p.name != plugin.name]

        self._plugins.append(plugin)
        self._plugins = self._sort(self._plugins)
        logger.info(
            "Registered plugin %s (priority=%d, enabled=%s)",
            plugin.name,
            plugin.priority,
            plugin.enabled,
        )

    # ── Lifecycle ──────────────────────────────────────────────────

    async def setup_all(self, config: EngineConfig) -> None:
        """
        Call :meth:`AbstractPlugin.setup` on every registered plugin
        (in priority order).  Failures on individual plugins are logged
        but do not halt the rest.
        """
        for plugin in self.active_plugins:
            try:
                await plugin.setup(config)
            except Exception as exc:
                logger.exception(
                    "Plugin %s.setup() failed: %s", plugin.name, exc
                )

    async def teardown_all(self) -> None:
        """Call :meth:`AbstractPlugin.teardown` on every plugin (reverse order)."""
        for plugin in reversed(self.active_plugins):
            try:
                await plugin.teardown()
            except Exception as exc:
                logger.exception(
                    "Plugin %s.teardown() failed: %s", plugin.name, exc
                )

    # ── Hook execution ─────────────────────────────────────────────

    async def run_pre_process(self, audio: bytes) -> bytes:
        """
        Run ``pre_process`` on each enabled plugin in priority order.

        The output of one plugin becomes the input of the next.  If a
        plugin returns empty bytes (``b\"\"``) the pipeline short-circuits
        and ``b\"\"`` is returned immediately.
        """
        for plugin in self.active_plugins:
            try:
                audio = await plugin.pre_process(audio)
            except Exception as exc:
                logger.exception(
                    "Plugin %s.pre_process() failed: %s", plugin.name, exc
                )
                continue
            if not audio:
                logger.info(
                    "Plugin %s.pre_process() returned empty — pipeline halted.",
                    plugin.name,
                )
                return b""
        return audio

    async def run_post_process(
        self, result: TranscriptResult
    ) -> TranscriptResult:
        """
        Run ``post_process`` on each enabled plugin in priority order.

        The output of one plugin becomes the input of the next.
        """
        for plugin in self.active_plugins:
            try:
                result = await plugin.post_process(result)
            except Exception as exc:
                logger.exception(
                    "Plugin %s.post_process() failed: %s", plugin.name, exc
                )
                continue
        return result

    # ── Accessors ──────────────────────────────────────────────────

    @property
    def plugins(self) -> list[AbstractPlugin]:
        """All registered plugins, sorted by priority (ascending)."""
        return list(self._plugins)

    @property
    def active_plugins(self) -> list[AbstractPlugin]:
        """Only the enabled plugins, sorted by priority (ascending)."""
        return [p for p in self._plugins if p.enabled]

    def get(self, name: str) -> AbstractPlugin | None:
        """Look up a plugin by its ``name``, or ``None`` if not found."""
        for plugin in self._plugins:
            if plugin.name == name:
                return plugin
        return None

    def count(self, *, only_enabled: bool = False) -> int:
        """Number of registered (or enabled) plugins."""
        if only_enabled:
            return len(self.active_plugins)
        return len(self._plugins)

    # ── Internals ──────────────────────────────────────────────────

    @staticmethod
    def _sort(plugins: list[AbstractPlugin]) -> list[AbstractPlugin]:
        return sorted(plugins, key=lambda p: (p.priority, p.name))

    def __repr__(self) -> str:
        return (
            f"PluginManager(count={self.count()}, "
            f"active={self.count(only_enabled=True)})"
        )

    def __len__(self) -> int:
        return len(self._plugins)

    def __iter__(self):  # type: ignore[return]
        return iter(self._plugins)

    def __getitem__(self, index: int) -> AbstractPlugin:
        return self._plugins[index]
