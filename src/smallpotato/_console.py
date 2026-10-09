import sys
from typing import TextIO

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
    print(line, file=output, flush=True)


class Progress:
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
        self._is_tty = self._stream.isatty()
        self._live_width = 0

    def update(
        self,
        current: int,
        /,
        *,
        remaining: float | None = None,
    ) -> None:
        if not 0 <= current <= self._total:
            raise ValueError(
                f"current must be between 0 and {self._total}, got {current}"
            )

        self._write(
            _format_progress(self._label, current, self._total, remaining)
        )

    def close(self) -> None:
        if self._is_tty and self._live_width:
            self._stream.write("\n")
            self._stream.flush()
            self._live_width = 0

    def _write(self, line: str) -> None:
        if self._is_tty:
            padding = " " * max(0, self._live_width - len(line))
            self._stream.write(f"\r{line}{padding}")
            self._live_width = len(line)
        else:
            self._stream.write(f"{line}\n")

        self._stream.flush()


def _format_progress(
    label: str,
    current: int,
    total: int,
    remaining: float | None,
) -> str:
    digits = len(str(total))
    line = (
        f"{label:<{_LABEL_WIDTH}}"
        f"{current:>{digits}}/{total}  "
        f"{current / total:6.1%}"
    )

    if current < total and remaining is not None:
        line += f"   eta {format_duration(remaining)}"

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
