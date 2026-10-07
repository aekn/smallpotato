import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

import pytest

from reproductions.rtmdet import _prepare, _record
from reproductions.rtmdet import reproduce as rtmdet


def test_new_run_dir_uses_stable_utc_names(tmp_path: Path) -> None:
    now = datetime(2026, 10, 6, 20, 30, 40, tzinfo=UTC)

    first = rtmdet.new_run_dir(tmp_path, now)
    second = rtmdet.new_run_dir(tmp_path, now)

    assert first.name == "20261006T203040Z"
    assert second.name == "20261006T203040Z-1"


def test_environment_isolates_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated/.venv")

    env = rtmdet.build_test_environment(tmp_path)

    assert env["PYTHONPATH"] == str(tmp_path)
    assert "VIRTUAL_ENV" not in env
    assert env["MPLBACKEND"] == "Agg"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_smoke_capture_requires_complete_output() -> None:
    capture = rtmdet.SmokeCapture()
    for line in (
        "python       3.10.22",
        "numpy        1.26.4",
        "torch        2.1.0+cu121",
        "torch cuda   12.1",
        "torchvision  0.16.0+cu121",
        "mmcv         2.1.0",
        "mmengine     0.10.4",
        "setuptools   69.5.1",
        "device       Tesla T4",
        "mmengine env ok",
        "mmcv ops     ok",
    ):
        capture.consume(line)

    result = capture.result()

    assert result["device"] == "Tesla T4"
    assert result["setuptools"] == "69.5.1"
    assert result["mmengine_env"] == "ok"


def test_upstream_output_parses_progress_and_warnings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = rtmdet.UpstreamOutput()

    output.consume("Epoch(test) [  50/1000]    eta: 0:06:25  time: 0.4053")
    output.consume("Epoch(test) [ 100/1000]    eta: 0:05:42  time: 0.3554")
    output.consume("UserWarning: historical warning")
    output.consume("Epoch(test) [1000/1000]    eta: 0:00:00  time: 0.3661")
    output.consume("Epoch(test) [1000/1000]    time: 0.3647")
    output.close()

    assert output.warning_count == 1
    assert capsys.readouterr().err.splitlines() == [
        "test          50/1000    5.0%   eta 6m25s",
        "test         100/1000   10.0%   eta 5m42s",
        "test        1000/1000  100.0%",
    ]


def test_reproduce_records_failed_test_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inputs = _prepare.PreparedInputs(
        source=source,
        source_fetched=False,
        config=source / "config.py",
        checkpoint=_prepare.AcquiredFile(tmp_path / "checkpoint.pth", None),
        coco=tmp_path / "coco",
        annotations=tmp_path / "instances_val2017.json",
        val_archive=_prepare.AcquiredFile(tmp_path / "val2017.zip", None),
        annotations_archive=_prepare.AcquiredFile(
            tmp_path / "annotations.zip", None
        ),
    )

    def prepare_environment(_log: TextIO) -> dict[str, str]:
        return {"device": "Tesla T4"}

    def prepare_inputs(_cache: Path) -> _prepare.PreparedInputs:
        return inputs

    def write_runtime_config(
        path: Path,
        _inputs: _prepare.PreparedInputs,
        /,
        *,
        prediction_prefix: Path,
        num_workers: int = 2,
    ) -> None:
        del prediction_prefix, num_workers
        path.write_text("test = True\n", encoding="utf-8")

    def build_provenance(**_kwargs: object) -> dict[str, object]:
        return {}

    def test_command(*_args: object) -> tuple[str, ...]:
        return ("test-command",)

    def fail_run_logged(*_args: object, **_kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, ("test-command",))

    monkeypatch.setattr(rtmdet, "_prepare_environment", prepare_environment)
    monkeypatch.setattr(_prepare, "prepare_inputs", prepare_inputs)
    monkeypatch.setattr(_prepare, "write_runtime_config", write_runtime_config)
    monkeypatch.setattr(_record, "build_provenance", build_provenance)
    monkeypatch.setattr(rtmdet, "_test_command", test_command)
    monkeypatch.setattr(rtmdet, "run_logged", fail_run_logged)

    runs = tmp_path / "runs"
    with pytest.raises(subprocess.CalledProcessError):
        rtmdet.reproduce(cache=tmp_path / "cache", runs=runs, num_workers=2)

    run = next(runs.iterdir())
    summary = cast(
        dict[str, object],
        json.loads((run / "summary.json").read_text(encoding="utf-8")),
    )
    durations = cast(dict[str, float], summary["durations_seconds"])

    assert summary["status"] == "error"
    assert "test" in durations
