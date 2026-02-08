"""Core multi-pane dashboard framework built on Rich."""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from demo_tui.spinners import Spinner


@dataclass
class PanelConfig:
    """Display configuration for a dashboard panel."""

    title: str = "Panel"
    border_style: str = "white"
    max_lines: int = 20


class KeyValuePanel:
    """Renders a dictionary of stats as aligned key-value lines."""

    def __init__(self, config: PanelConfig | None = None) -> None:
        self.config = config or PanelConfig()
        self._data: OrderedDict[str, str] = OrderedDict()

    def set(self, key: str, value: str) -> None:
        """Set or update a key-value pair."""
        self._data[key] = value

    def render(self) -> Panel:
        """Return a Rich Panel with aligned key-value lines."""
        if not self._data:
            body = "(empty)"
        else:
            max_key_len = max(len(k) for k in self._data)
            lines: list[str] = []
            for k, v in self._data.items():
                lines.append(f"{k:<{max_key_len}}  {v}")
            body = "\n".join(lines[: self.config.max_lines])
        return Panel(
            Text(body, overflow="fold"),
            title=self.config.title,
            border_style=self.config.border_style,
        )


class LogPanel:
    """Renders a scrolling log tail backed by a deque."""

    def __init__(self, config: PanelConfig | None = None, max_lines: int = 200) -> None:
        self.config = config or PanelConfig(title="Logs", border_style="magenta")
        self._lines: deque[str] = deque(maxlen=max_lines)
        self._tail_size: int = self.config.max_lines

    def log(self, message: str) -> None:
        """Append a timestamped message."""
        ts = time.strftime("%H:%M:%S")
        self._lines.append(f"[{ts}] {message}")

    def render(self) -> Panel:
        """Return a Rich Panel showing the most recent log lines."""
        tail = list(self._lines)[-max(self._tail_size, 1) :]
        body = "\n".join(tail) if tail else "No logs yet"
        return Panel(
            Text(body, overflow="fold"),
            title=self.config.title,
            border_style=self.config.border_style,
        )


class ProgressPanel:
    """Renders a progress bar with spinner and ETA."""

    def __init__(self, config: PanelConfig | None = None) -> None:
        self.config = config or PanelConfig(title="Progress", border_style="green")
        self._current: int = 0
        self._total: int = 1
        self._extra_text: str = ""
        self._start_time: float = time.perf_counter()
        self._spinner = Spinner()

    def update(self, current: int, total: int, extra_text: str = "") -> None:
        """Update progress state."""
        self._current = current
        self._total = max(total, 1)
        self._extra_text = extra_text

    def render(self) -> Panel:
        """Return a Rich Panel with progress bar, spinner, and ETA."""
        pct = (self._current / self._total) * 100.0
        elapsed = time.perf_counter() - self._start_time

        # ETA calculation
        if self._current > 0:
            remaining = self._total - self._current
            rate = elapsed / self._current
            eta = remaining * rate
            eta_str = f"{eta:.1f}s"
        else:
            eta_str = "..."

        # Text-based progress bar (30 chars wide)
        bar_width = 30
        filled = int(round(bar_width * self._current / self._total))
        bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

        spin_char = self._spinner.next()
        lines = [
            f"{spin_char} [{bar}] {pct:>5.1f}%  ({self._current}/{self._total})",
            f"  elapsed={elapsed:.1f}s  eta={eta_str}",
        ]
        if self._extra_text:
            lines.append(f"  {self._extra_text}")

        return Panel(
            Text("\n".join(lines), overflow="fold"),
            title=self.config.title,
            border_style=self.config.border_style,
        )


class ComparisonPanel:
    """Side-by-side two-column output using Rich Layout."""

    def __init__(self, config: PanelConfig | None = None) -> None:
        self.config = config or PanelConfig(title="Comparison", border_style="blue")
        self._left_lines: list[str] = []
        self._right_lines: list[str] = []
        self._left_title: str = "Left"
        self._right_title: str = "Right"

    def set_left(self, lines: list[str], title: str | None = None) -> None:
        """Set the left column content."""
        self._left_lines = list(lines)
        if title is not None:
            self._left_title = title

    def set_right(self, lines: list[str], title: str | None = None) -> None:
        """Set the right column content."""
        self._right_lines = list(lines)
        if title is not None:
            self._right_title = title

    def render(self) -> Panel:
        """Return a Rich Panel with two side-by-side columns."""
        left_text = "\n".join(self._left_lines) if self._left_lines else "(empty)"
        right_text = "\n".join(self._right_lines) if self._right_lines else "(empty)"

        inner = Layout()
        inner.split_row(
            Layout(
                Panel(
                    Text(left_text, overflow="fold"),
                    title=self._left_title,
                    border_style="dim",
                ),
                name="left",
            ),
            Layout(
                Panel(
                    Text(right_text, overflow="fold"),
                    title=self._right_title,
                    border_style="dim",
                ),
                name="right",
            ),
        )
        return Panel(
            inner,
            title=self.config.title,
            border_style=self.config.border_style,
        )


class DashboardBase:
    """Base class for multi-pane Rich dashboards.

    Register named panels, then call ``run()`` to drive a Live display loop.
    """

    def __init__(self, title: str = "Dashboard", refresh_hz: float = 4.0) -> None:
        self.title = title
        self.refresh_hz = refresh_hz
        self.console = Console()
        self.start_time: float = time.perf_counter()
        self._panels: OrderedDict[str, tuple[PanelConfig, Any]] = OrderedDict()

    def add_panel(self, name: str, panel: Any, config: PanelConfig | None = None) -> None:
        """Register a named panel.

        If *config* is ``None``, a default ``PanelConfig(title=name)`` is used.
        The panel object must expose a ``render() -> Panel`` method.
        """
        if config is None:
            # Try to use the panel's own config if it has one
            cfg = getattr(panel, "config", None)
            if isinstance(cfg, PanelConfig):
                config = cfg
            else:
                config = PanelConfig(title=name)
        self._panels[name] = (config, panel)

    def log(self, message: str) -> None:
        """Log a message. Delegates to the 'log' panel if it is a LogPanel."""
        if "log" in self._panels:
            _, panel = self._panels["log"]
            if isinstance(panel, LogPanel):
                panel.log(message)
                return
        self.console.print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def set_status(self, panel_name: str, key: str, value: str) -> None:
        """Set a key-value pair on a KeyValuePanel by name."""
        if panel_name in self._panels:
            _, panel = self._panels[panel_name]
            if isinstance(panel, KeyValuePanel):
                panel.set(key, value)

    def render(self) -> Layout:
        """Build a Layout from all registered panels.

        Non-log panels go in the top row (split equally); a panel named
        "log" (if present and a LogPanel) occupies the bottom row.
        """
        top_panels: list[tuple[str, PanelConfig, Any]] = []
        log_entry: tuple[str, PanelConfig, Any] | None = None

        for name, (cfg, panel) in self._panels.items():
            if name == "log" and isinstance(panel, LogPanel):
                log_entry = (name, cfg, panel)
            else:
                top_panels.append((name, cfg, panel))

        layout = Layout(name="root")

        if log_entry is not None and top_panels:
            # Two-row layout: top for stat panels, bottom for logs
            layout.split_column(
                Layout(name="top", size=16),
                Layout(name="logs"),
            )
            top_layout = layout["top"]
            top_children = []
            for pname, _cfg, panel in top_panels:
                child = Layout(name=pname)
                child.update(panel.render())
                top_children.append(child)
            top_layout.split_row(*top_children)

            _, _log_cfg, log_panel = log_entry
            layout["logs"].update(log_panel.render())

        elif log_entry is not None:
            # Only a log panel
            _, _log_cfg, log_panel = log_entry
            layout.update(log_panel.render())

        elif top_panels:
            # Only top panels, no log
            top_children = []
            for pname, _cfg, panel in top_panels:
                child = Layout(name=pname)
                child.update(panel.render())
                top_children.append(child)
            layout.split_row(*top_children)

        else:
            layout.update(Panel("No panels registered"))

        return layout

    def run(self, callback: Callable[[DashboardBase], None], refresh_hz: float | None = None) -> None:
        """Enter a Rich Live context and invoke *callback(self)*.

        The callback receives this dashboard instance and is responsible for
        calling ``live.update(self.render())`` at the appropriate cadence.
        The Live object is stored as ``self._live`` during execution so that
        the callback (or methods called by it) can trigger manual updates.
        """
        hz = refresh_hz if refresh_hz is not None else self.refresh_hz
        with Live(
            self.render(),
            console=self.console,
            refresh_per_second=max(hz, 1.0),
            screen=self.console.is_terminal,
            transient=False,
        ) as live:
            self._live = live
            try:
                callback(self)
            finally:
                self._live = None  # type: ignore[assignment]

    def update_live(self) -> None:
        """Push a fresh render to the Live display (safe to call outside ``run``)."""
        live = getattr(self, "_live", None)
        if live is not None:
            live.update(self.render())
