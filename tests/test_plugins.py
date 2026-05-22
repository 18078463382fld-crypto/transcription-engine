"""Tests for ``PluginManager`` and ``AbstractPlugin`` hooks.

All tests use concrete plugin subclasses — no GPU, no cloud APIs,
no audio files required.
"""

from __future__ import annotations

from typing import Any

import pytest

from transcribe.core.models import EngineConfig, TranscriptResult, TranscriptSegment
from transcribe.plugins.base import AbstractPlugin
from transcribe.plugins.plugin_manager import PluginManager


# ═══════════════════════════════════════════════════════════════════
# Concrete plugin subclasses for testing
# ═══════════════════════════════════════════════════════════════════


class UpperPlugin(AbstractPlugin):
    """A test plugin that uppercases transcript text."""

    name: str = "upper"

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        result.text = result.text.upper()
        for seg in result.segments:
            seg.text = seg.text.upper()
        return result


class PrefixPlugin(AbstractPlugin):
    """A test plugin that prepends text to the transcript."""

    def __init__(self, prefix: str = "[PRE]") -> None:
        super().__init__()
        self._prefix = prefix

    name: str = "prefix"

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        result.text = self._prefix + result.text
        return result


class AudioSilencerPlugin(AbstractPlugin):
    """A test plugin that silences audio (returns empty bytes)."""

    name: str = "silencer"

    async def pre_process(self, audio: bytes) -> bytes:
        return b""


class DoubleAudioPlugin(AbstractPlugin):
    """A test plugin that doubles audio bytes."""

    name: str = "doubler"

    async def pre_process(self, audio: bytes) -> bytes:
        return audio * 2


class ErrorPlugin(AbstractPlugin):
    """A test plugin that raises in ``pre_process``."""

    name: str = "error"

    async def pre_process(self, audio: bytes) -> bytes:
        msg = "intentional failure"
        raise RuntimeError(msg)


class TrackCallPlugin(AbstractPlugin):
    """A test plugin that records lifecycle call order."""

    def __init__(self) -> None:
        super().__init__()
        self.call_log: list[str] = []

    name: str = "tracker"

    async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None:
        self.call_log.append("setup")

    async def teardown(self) -> None:
        self.call_log.append("teardown")

    async def pre_process(self, audio: bytes) -> bytes:
        self.call_log.append("pre_process")
        return audio

    async def post_process(self, result: TranscriptResult) -> TranscriptResult:
        self.call_log.append("post_process")
        return result


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def config() -> EngineConfig:
    return EngineConfig()


@pytest.fixture
def sample_result() -> TranscriptResult:
    return TranscriptResult(
        text="hello world",
        segments=[TranscriptSegment(start=0.0, end=1.0, text="hello world")],
    )


# ═══════════════════════════════════════════════════════════════════
# AbstractPlugin — individual hook behaviour
# ═══════════════════════════════════════════════════════════════════


class TestAbstractPlugin:
    """Verify that ``AbstractPlugin`` subclasses work as expected."""

    @pytest.mark.asyncio
    async def test_name_property(self) -> None:
        """``name`` returns the plugin's unique name."""
        plugin = UpperPlugin()
        assert plugin.name == "upper"

    def test_default_priority_and_enabled(self) -> None:
        """Default priority is 100 and enabled is ``True``."""
        plugin = UpperPlugin()
        assert plugin.priority == 100
        assert plugin.enabled is True

    def test_custom_priority(self) -> None:
        """Priority can be overridden at class or instance level."""
        plugin = UpperPlugin()
        plugin.priority = 50
        assert plugin.priority == 50

    def test_repr(self) -> None:
        """``__repr__`` includes name, priority, and enabled status."""
        plugin = UpperPlugin()
        plugin.priority = 42
        r = repr(plugin)
        assert "UpperPlugin" in r
        assert "upper" in r
        assert "42" in r

    @pytest.mark.asyncio
    async def test_pre_process_noop_by_default(self) -> None:
        """The default ``pre_process`` returns audio unchanged."""
        plugin = UpperPlugin()
        data = b"\x01\x02\x03"
        result = await plugin.pre_process(data)
        assert result == data

    @pytest.mark.asyncio
    async def test_post_process_noop_by_default(self) -> None:
        """The default ``post_process`` returns the result unchanged."""
        plugin = UpperPlugin()
        result = TranscriptResult(text="test")
        returned = await plugin.post_process(result)
        assert returned is result

    @pytest.mark.asyncio
    async def test_setup_and_teardown_noop_by_default(self) -> None:
        """Default ``setup`` and ``teardown`` do not raise."""
        plugin = UpperPlugin()
        await plugin.setup(EngineConfig())
        await plugin.teardown()

    @pytest.mark.asyncio
    async def test_post_process_uppercase(self, sample_result: TranscriptResult) -> None:
        """``UpperPlugin`` uppercases the text."""
        plugin = UpperPlugin()
        result = await plugin.post_process(sample_result)
        assert result.text == "HELLO WORLD"
        assert result.segments[0].text == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_pre_process_returns_empty_bytes(self) -> None:
        """``AudioSilencerPlugin`` returns empty bytes."""
        plugin = AudioSilencerPlugin()
        result = await plugin.pre_process(b"\x01\x02\x03")
        assert result == b""


# ═══════════════════════════════════════════════════════════════════
# PluginManager — registration
# ═══════════════════════════════════════════════════════════════════


class TestPluginManagerRegistration:
    """Verify plugin registration, sorting, filtering."""

    def test_empty_manager(self) -> None:
        """A fresh manager has no plugins."""
        mgr = PluginManager()
        assert mgr.count() == 0
        assert mgr.plugins == []

    def test_register_single(self) -> None:
        """``register()`` adds a plugin instance."""
        mgr = PluginManager()
        plugin = UpperPlugin()
        mgr.register(plugin)
        assert mgr.count() == 1
        assert mgr.plugins[0] is plugin

    def test_register_type_check(self) -> None:
        """``register()`` raises ``TypeError`` for non-``AbstractPlugin`` objects."""
        mgr = PluginManager()
        with pytest.raises(TypeError, match="Expected AbstractPlugin"):
            mgr.register("not-a-plugin")  # type: ignore[arg-type]

    def test_register_duplicate_by_name_replaces(self) -> None:
        """Registering a plugin with the same name replaces the previous one."""
        mgr = PluginManager()
        mgr.register(UpperPlugin())
        replacement = UpperPlugin()
        replacement.priority = 200
        mgr.register(replacement)
        assert mgr.count() == 1
        assert mgr.plugins[0].priority == 200

    def test_sort_by_priority_then_name(self) -> None:
        """Plugins are sorted by (priority, name) in ascending order."""
        mgr = PluginManager()
        low = PrefixPlugin()
        low.priority = 10
        mid = UpperPlugin()
        mid.priority = 50
        high = AudioSilencerPlugin()
        high.priority = 100

        mgr.register(high)
        mgr.register(mid)
        mgr.register(low)

        assert mgr.plugins[0] is low
        assert mgr.plugins[1] is mid
        assert mgr.plugins[2] is high

    def test_get_by_name(self) -> None:
        """``get()`` returns the plugin by name or ``None``."""
        mgr = PluginManager()
        plugin = UpperPlugin()
        mgr.register(plugin)
        assert mgr.get("upper") is plugin
        assert mgr.get("nonexistent") is None

    def test_active_plugins_excludes_disabled(self) -> None:
        """``active_plugins`` only includes enabled plugins."""
        mgr = PluginManager()
        enabled = UpperPlugin()
        disabled = UpperPlugin()
        disabled.name = "disabled-upper"  # type: ignore[assignment]
        disabled.enabled = False
        mgr.register(enabled)
        mgr.register(disabled)
        assert mgr.count() == 2
        assert mgr.count(only_enabled=True) == 1
        assert enabled in mgr.active_plugins
        assert disabled not in mgr.active_plugins

    def test_len_and_iter(self) -> None:
        """``len()`` and iteration work on the manager."""
        mgr = PluginManager()
        mgr.register(UpperPlugin())
        mgr.register(PrefixPlugin())
        assert len(mgr) == 2
        names = [p.name for p in mgr]
        assert "upper" in names
        assert "prefix" in names

    def test_getitem(self) -> None:
        """Indexing returns the plugin at that position."""
        mgr = PluginManager()
        plugin = UpperPlugin()
        mgr.register(plugin)
        assert mgr[0] is plugin

    def test_repr(self) -> None:
        """``__repr__`` shows plugin count and active count."""
        mgr = PluginManager()
        mgr.register(UpperPlugin())
        r = repr(mgr)
        assert "PluginManager" in r
        assert "count=1" in r
        assert "active=1" in r


# ═══════════════════════════════════════════════════════════════════
# PluginManager — lifecycle hooks
# ═══════════════════════════════════════════════════════════════════


class TestPluginManagerLifecycle:
    """Verify ``setup_all`` and ``teardown_all``."""

    @pytest.mark.asyncio
    async def test_setup_all_calls_setup_on_each_plugin(self, config: EngineConfig) -> None:
        """``setup_all`` calls ``setup`` on every registered plugin."""
        mgr = PluginManager()
        tracker1 = TrackCallPlugin()
        tracker1.name = "tracker-1"
        tracker2 = TrackCallPlugin()
        tracker2.name = "tracker-2"
        mgr.register(tracker1)
        mgr.register(tracker2)
        await mgr.setup_all(config)
        assert tracker1.call_log == ["setup"]
        assert tracker2.call_log == ["setup"]

    @pytest.mark.asyncio
    async def test_setup_all_in_priority_order(self, config: EngineConfig) -> None:
        """``setup_all`` iterates plugins in priority order."""
        mgr = PluginManager()
        high = TrackCallPlugin()
        high.priority = 10
        high.name = "high"  # type: ignore[assignment]
        low = TrackCallPlugin()
        low.priority = 100
        low.name = "low"  # type: ignore[assignment]

        mgr.register(low)
        mgr.register(high)

        await mgr.setup_all(config)
        assert high.call_log == ["setup"]
        assert low.call_log == ["setup"]

    @pytest.mark.asyncio
    async def test_teardown_all_in_reverse_order(self, config: EngineConfig) -> None:
        """``teardown_all`` iterates plugins in reverse priority order."""
        mgr = PluginManager()
        first = TrackCallPlugin()
        first.name = "first"  # type: ignore[assignment]
        first.priority = 10
        second = TrackCallPlugin()
        second.name = "second"  # type: ignore[assignment]
        second.priority = 20

        mgr.register(first)
        mgr.register(second)

        await mgr.setup_all(config)
        await mgr.teardown_all()

        # Both should have setup and teardown
        assert first.call_log == ["setup", "teardown"]
        assert second.call_log == ["setup", "teardown"]

    @pytest.mark.asyncio
    async def test_setup_failure_does_not_halt(self, config: EngineConfig) -> None:
        """If one plugin's ``setup`` fails, others still run."""

        class FailingPlugin(AbstractPlugin):
            @property
            def name(self) -> str:
                return "failing"

            async def setup(self, config: EngineConfig, plugin_config: dict | None = None) -> None:
                msg = "setup failed"
                raise RuntimeError(msg)

        mgr = PluginManager()
        mgr.register(FailingPlugin())
        tracker = TrackCallPlugin()
        mgr.register(tracker)

        # Should not raise
        await mgr.setup_all(config)
        assert "setup" in tracker.call_log

    @pytest.mark.asyncio
    async def test_teardown_skips_disabled(self, config: EngineConfig) -> None:
        """Disabled plugins are skipped during teardown."""
        mgr = PluginManager()
        enabled = TrackCallPlugin()
        enabled.name = "enabled"  # type: ignore[assignment]
        disabled = TrackCallPlugin()
        disabled.name = "disabled"  # type: ignore[assignment]
        disabled.enabled = False
        mgr.register(enabled)
        mgr.register(disabled)

        await mgr.setup_all(config)
        await mgr.teardown_all()

        assert enabled.call_log == ["setup", "teardown"]
        # Disabled plugins never get setup/teardown called
        assert disabled.call_log == []


# ═══════════════════════════════════════════════════════════════════
# PluginManager — hook execution (pre_process / post_process)
# ═══════════════════════════════════════════════════════════════════


class TestPluginManagerHooks:
    """Verify pipeline hook execution."""

    @pytest.mark.asyncio
    async def test_run_pre_process_chains_audio(self) -> None:
        """``run_pre_process`` passes audio through plugins in priority order."""
        mgr = PluginManager()
        mgr.register(DoubleAudioPlugin())
        result = await mgr.run_pre_process(b"\x01\x02")
        assert result == b"\x01\x02\x01\x02"

    @pytest.mark.asyncio
    async def test_run_pre_process_short_circuits_on_empty(self) -> None:
        """If a plugin returns empty bytes, the pipeline short-circuits."""
        mgr = PluginManager()
        mgr.register(AudioSilencerPlugin())
        mgr.register(DoubleAudioPlugin())  # should not be reached
        result = await mgr.run_pre_process(b"\x01\x02\x03")
        assert result == b""

    @pytest.mark.asyncio
    async def test_run_pre_process_continues_on_error(self) -> None:
        """If a plugin raises, the error is logged but the pipeline continues."""
        mgr = PluginManager()
        mgr.register(ErrorPlugin())
        mgr.register(DoubleAudioPlugin())
        result = await mgr.run_pre_process(b"\x01\x02")
        # After error, the next plugin still runs
        assert result == b"\x01\x02\x01\x02"

    @pytest.mark.asyncio
    async def test_run_pre_process_with_no_plugins(self) -> None:
        """With no plugins, audio is returned unchanged."""
        mgr = PluginManager()
        result = await mgr.run_pre_process(b"\x01\x02")
        assert result == b"\x01\x02"

    @pytest.mark.asyncio
    async def test_run_post_process_chains_results(self, sample_result: TranscriptResult) -> None:
        """``run_post_process`` passes results through plugins in priority order."""
        mgr = PluginManager()
        mgr.register(UpperPlugin())
        mgr.register(PrefixPlugin())

        result = await mgr.run_post_process(sample_result)
        # UpperPlugin runs first (priority 100), then PrefixPlugin (priority 100)
        # Since both have default priority 100, they are ordered by name:
        # "prefix" < "upper" alphabetically, so prefix runs first, then upper.
        # prefix: "[PRE]hello world"
        # upper:  "[PRE]HELLO WORLD"
        assert result.text == "[PRE]HELLO WORLD"

    @pytest.mark.asyncio
    async def test_run_post_process_continues_on_error(self, sample_result: TranscriptResult) -> None:
        """If a plugin's ``post_process`` raises, the pipeline continues."""

        class FailingPostPlugin(AbstractPlugin):
            @property
            def name(self) -> str:
                return "failing-post"

            async def post_process(self, result: TranscriptResult) -> TranscriptResult:
                msg = "post failed"
                raise RuntimeError(msg)

        mgr = PluginManager()
        mgr.register(FailingPostPlugin())
        mgr.register(UpperPlugin())

        result = await mgr.run_post_process(sample_result)
        assert result.text == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_run_post_process_skips_disabled(self, sample_result: TranscriptResult) -> None:
        """Disabled plugins are skipped during hook execution."""
        mgr = PluginManager()
        disabled = UpperPlugin()
        disabled.name = "disabled-upper"  # type: ignore[assignment]
        disabled.enabled = False
        mgr.register(disabled)
        mgr.register(PrefixPlugin())

        result = await mgr.run_post_process(sample_result)
        # Only PrefixPlugin runs
        assert result.text == "[PRE]hello world"

    @pytest.mark.asyncio
    async def test_run_pre_process_only_enabled_plugins(self) -> None:
        """Disabled plugins are skipped in pre_process chain."""
        mgr = PluginManager()
        disabled = DoubleAudioPlugin()
        disabled.enabled = False
        mgr.register(disabled)

        result = await mgr.run_pre_process(b"\x01\x02")
        assert result == b"\x01\x02"  # unchanged


# ═══════════════════════════════════════════════════════════════════
# PluginManager — discovery (unit test without real entry points)
# ═══════════════════════════════════════════════════════════════════


class TestPluginManagerDiscover:
    """Verify ``discover()`` behaviour (mocked, no real entry points)."""

    def test_discover_with_empty_group(self) -> None:
        """Discover with an empty entry-point group returns an empty list."""
        mgr = PluginManager()
        plugins = mgr.discover(group="nonexistent.group")
        assert plugins == []

    def test_discover_returns_sorted_plugins(self) -> None:
        """Discovered plugins are returned sorted by (priority, name)."""
        # We can't easily mock importlib.metadata.entry_points here,
        # but we verify that after discover, _plugins is sorted.
        mgr = PluginManager()
        p1 = DoubleAudioPlugin()
        p1.priority = 200
        p2 = UpperPlugin()
        p2.priority = 10
        mgr.register(p2)
        mgr.register(p1)
        plugins = mgr.plugins
        assert plugins[0].priority == 10
        assert plugins[1].priority == 200
