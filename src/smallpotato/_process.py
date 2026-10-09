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
        start_new_session=os.name == "posix",
    ) as process:
        stream = process.stdout
        assert stream is not None

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

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, args)


def run_captured(
    args: Sequence[str],
    /,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr and (stderr := exc.stderr.strip()):
            exc.add_note(stderr)
        raise

    return result.stdout


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
