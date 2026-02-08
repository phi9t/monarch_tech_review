"""Spinner animation helpers for terminal UIs."""

from __future__ import annotations


class Spinner:
    """Cycles through a sequence of characters for animated spinners."""

    BRAILLE: list[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    DOTS: list[str] = ["⠄", "⠂", "⠁", "⠂", "⠄", "⠠", "⠐", "⠈"]
    ARROWS: list[str] = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]

    def __init__(self, chars: str | list[str] | None = None) -> None:
        if chars is None:
            self._chars = list(self.BRAILLE)
        elif isinstance(chars, str):
            self._chars = list(chars)
        else:
            self._chars = list(chars)
        if not self._chars:
            self._chars = list(self.BRAILLE)
        self._idx: int = 0

    def next(self) -> str:
        """Return the current character and advance to the next."""
        ch = self._chars[self._idx]
        self._idx = (self._idx + 1) % len(self._chars)
        return ch

    def current(self) -> str:
        """Return the current character without advancing."""
        return self._chars[self._idx]
