__all__ = ()

import os
import signal
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TextIO

_TERMINATE_TIMEOUT = 5.0


def run_logged(
    args: Sequence[str],
    /,
    *,
    log: TextIO,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
) -> None:
    """Run *args* and write combined output to *log*."""
    if not args:
        raise ValueError("args must not be empty")

    with subprocess.Popen(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=os.name == "posix",
    ) as process:
        stream = process.stdout
        if stream is None:
            raise RuntimeError("stdout pipe is unavailable")

        try:
            for line in stream:
                log.write(line)
                log.flush()
                if on_line is not None:
                    on_line(line.removesuffix("\n"))

            returncode = process.wait()
        except BaseException:
            _terminate(process)
            raise

    if returncode:
        raise subprocess.CalledProcessError(returncode, args)


def capture_output(
    args: Sequence[str],
    /,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Run *args* and return stripped stdout."""
    if not args:
        raise ValueError("args must not be empty")

    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode:
        error = subprocess.CalledProcessError(
            result.returncode,
            args,
            output=result.stdout,
            stderr=result.stderr,
        )
        if stderr := result.stderr.strip():
            error.add_note(stderr)
        raise error
    return result.stdout.strip()


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=_TERMINATE_TIMEOUT)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass

    process.wait()
