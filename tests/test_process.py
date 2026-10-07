import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smallpotato._process import capture_output, run_logged


def test_capture_output_returns_stripped_stdout(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["SMALLPOTATO_TEST"] = "value"
    code = (
        "import os; print(os.getcwd()); print(os.environ['SMALLPOTATO_TEST'])"
    )

    output = capture_output(
        (sys.executable, "-c", code),
        cwd=tmp_path,
        env=env,
    )

    assert output.splitlines() == [str(tmp_path), "value"]


def test_capture_output_raises_on_nonzero_exit() -> None:
    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        capture_output(
            (
                sys.executable,
                "-c",
                "import sys; print('bad', file=sys.stderr); "
                "raise SystemExit(9)",
            )
        )

    assert exc_info.value.returncode == 9
    assert exc_info.value.stderr == "bad\n"


def test_run_logged_streams_combined_output() -> None:
    log = io.StringIO()
    lines: list[str] = []
    code = (
        "import sys; "
        "print('stdout', flush=True); "
        "print('stderr', file=sys.stderr, flush=True)"
    )

    run_logged(
        (sys.executable, "-c", code),
        log=log,
        on_line=lines.append,
    )

    assert log.getvalue() == "stdout\nstderr\n"
    assert lines == ["stdout", "stderr"]


def test_run_logged_uses_cwd_and_environment(tmp_path: Path) -> None:
    log = io.StringIO()
    env = os.environ.copy()
    env["SMALLPOTATO_TEST"] = "value"
    code = (
        "import os; print(os.getcwd()); print(os.environ['SMALLPOTATO_TEST'])"
    )

    run_logged(
        (sys.executable, "-c", code),
        log=log,
        cwd=tmp_path,
        env=env,
    )

    assert log.getvalue().splitlines() == [str(tmp_path), "value"]


def test_run_logged_raises_on_nonzero_exit() -> None:
    log = io.StringIO()

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        run_logged(
            (sys.executable, "-c", "raise SystemExit(7)"),
            log=log,
        )

    assert exc_info.value.returncode == 7


def test_run_logged_stops_process_when_callback_fails() -> None:
    log = io.StringIO()
    code = "import time; print('ready', flush=True); time.sleep(30)"

    def fail(_: str) -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        run_logged(
            (sys.executable, "-c", code),
            log=log,
            on_line=fail,
        )

    assert log.getvalue() == "ready\n"
