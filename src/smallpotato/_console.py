__all__ = ()

import sys
from typing import Self, TextIO, final

_LABEL_WIDTH = 12


def write_status(
    label: str,
    text: str = "",
    /,
    *,
    stream: TextIO | None = None,
) -> None:
    output = sys.stderr if stream is None else stream
    line = f"{label:<{_LABEL_WIDTH}}{text}".rstrip()
    output.write(f"{line}\n")
    output.flush()


@final
class Progress:
    """Report bounded progress to a terminal or text stream."""

    __slots__ = ("_label", "_live_width", "_stream", "_total", "_tty")

    def __init__(
        self,
        label: str,
        total: int,
        /,
        *,
        stream: TextIO | None = None,
    ) -> None:
        if total <= 0:
            raise ValueError("total must be positive")

        self._label = label
        self._total = total
        self._stream = sys.stderr if stream is None else stream
        self._tty = self._stream.isatty()
        self._live_width = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def update(
        self,
        current: int,
        /,
        *,
        remaining: float | None = None,
    ) -> None:
        if current < 0 or current > self._total:
            raise ValueError(
                f"current must be between 0 and {self._total}, got {current}"
            )
        if remaining is not None and remaining < 0:
            raise ValueError("remaining must not be negative")

        self._write(
            _progress_line(
                self._label,
                current,
                self._total,
                remaining,
            )
        )

    def close(self) -> None:
        if self._tty and self._live_width:
            self._stream.write("\n")
            self._stream.flush()
            self._live_width = 0

    def _write(self, line: str) -> None:
        if self._tty:
            padding = " " * max(0, self._live_width - len(line))
            self._stream.write(f"\r{line}{padding}")
            self._live_width = len(line)
        else:
            self._stream.write(f"{line}\n")
        self._stream.flush()


def _progress_line(
    label: str,
    current: int,
    total: int,
    remaining: float | None,
) -> str:
    digits = len(str(total))
    percent = 100.0 * current / total
    line = (
        f"{label:<{_LABEL_WIDTH}}"
        f"{current:>{digits}}/{total:<{digits}}  "
        f"{percent:5.1f}%"
    )

    if current != total and remaining is not None:
        return f"{line}   eta {format_duration(remaining)}"
    return line


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"
