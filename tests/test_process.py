import io
import sys

import pytest

from smallpotato._process import run_logged


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
