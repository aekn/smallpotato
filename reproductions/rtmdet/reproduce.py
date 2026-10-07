__all__ = ()

import argparse
import os
import re
import shlex
import shutil
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TextIO, cast, final

from reproductions.rtmdet import _prepare, _record

from smallpotato._console import Progress, format_duration, write_status
from smallpotato._process import capture_output, run_logged
from smallpotato.evaluation import COCOMetrics, eval_coco

_ROOT = Path(__file__).resolve().parents[2]
_ENV = Path(__file__).with_name("env")
_EXPECTED = Path(__file__).with_name("expected.json")
_SMOKE = Path(__file__).with_name("smoke.py")
_TEST_PROGRESS = re.compile(
    r"\bEpoch\(test\)\s+\[\s*(\d+)/(\d+)\]"
    r"(?:\s+eta:\s*(\d+):(\d+):(\d+))?"
)
_WARNING = re.compile(r"\bWARNING\b|\b[A-Za-z]*Warning:")
_SMOKE_KEYS = (
    ("python", "python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torch cuda", "torch_cuda"),
    ("torchvision", "torchvision"),
    ("mmcv", "mmcv"),
    ("mmengine", "mmengine"),
    ("setuptools", "setuptools"),
    ("device", "device"),
    ("mmengine env", "mmengine_env"),
    ("mmcv ops", "mmcv_ops"),
)


@final
class UpstreamOutput:
    __slots__ = ("_last_progress", "_progress", "warning_count")

    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._last_progress = -1
        self.warning_count = 0

    def consume(self, line: str, /) -> None:
        match = _TEST_PROGRESS.search(line)
        if match is not None:
            current = int(match.group(1))
            total = int(match.group(2))
            if current > self._last_progress:
                if self._progress is None:
                    self._progress = Progress("test", total)
                remaining = _remaining_seconds(match)
                self._progress.update(current, remaining=remaining)
                self._last_progress = current

        if _WARNING.search(line) is not None:
            self.warning_count += 1

    def close(self) -> None:
        if self._progress is not None:
            self._progress.close()


def _remaining_seconds(match: re.Match[str]) -> int | None:
    hours, minutes, seconds = match.group(3, 4, 5)
    if hours is None or minutes is None or seconds is None:
        return None
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


@final
class SmokeCapture:
    __slots__ = ("values",)

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def consume(self, line: str, /) -> None:
        for prefix, key in _SMOKE_KEYS:
            token = f"{prefix:<13}"
            if line.startswith(token):
                self.values[key] = line[len(token) :].strip()
                return

    def result(self) -> dict[str, str]:
        missing = {key for _, key in _SMOKE_KEYS} - self.values.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"incomplete environment smoke output: {names}")
        return dict(self.values)


@final
class _PhaseTimer:
    __slots__ = ("_active", "_durations", "_started")

    def __init__(self) -> None:
        self._active: str | None = None
        self._durations: dict[str, float] = {}
        self._started = perf_counter()

    def begin(self, phase: str, /) -> None:
        now = perf_counter()
        if self._active is not None:
            self._durations[self._active] = now - self._started
        self._active = phase
        self._started = now
        write_status("coco", phase)

    def finish(self) -> dict[str, float]:
        now = perf_counter()
        if self._active is not None:
            self._durations[self._active] = now - self._started
            self._active = None
        return dict(self._durations)


def main(argv: list[str] | None = None) -> int:
    cache, runs, num_workers = _parse_args(argv)
    try:
        return reproduce(cache=cache, runs=runs, num_workers=num_workers)
    except KeyboardInterrupt:
        return 130


def reproduce(*, cache: Path, runs: Path, num_workers: int) -> int:
    """Run the RTMDet-tiny COCO reproduction."""
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    started_at = datetime.now(UTC)
    started = perf_counter()
    run = new_run_dir(runs, started_at)
    log_path = run / "run.log"
    summary_path = run / "summary.json"
    provenance_path = run / "provenance.json"
    runtime_path = run / "runtime.py"
    prediction_path = run / "pred.bbox.json"
    upstream = run / "upstream"
    durations: dict[str, float] = {}

    _write_header(run)

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        _log(log, f"started {_record.iso_utc(started_at)}")
        try:
            env_started = perf_counter()
            environment = _prepare_environment(log)
            durations["environment"] = perf_counter() - env_started
            write_status("device", environment["device"])
            _write_check("environment")

            inputs_started = perf_counter()
            write_status("prepare", "inputs")
            inputs = _prepare.prepare_inputs(cache)
            durations["inputs"] = perf_counter() - inputs_started
            _log_inputs(log, inputs)
            _write_input_checks()

            _prepare.write_runtime_config(
                runtime_path,
                inputs,
                prediction_prefix=run / "pred",
                num_workers=num_workers,
            )
            upstream.mkdir()

            command = _test_command(inputs, runtime_path, upstream)
            provenance = _record.build_provenance(
                root=_ROOT,
                expected=_EXPECTED,
                environment_lock=_ENV / "uv.lock",
                started_at=started_at,
                cache=cache,
                environment=environment,
                command=command,
                num_workers=num_workers,
                runtime=runtime_path,
                inputs=inputs,
            )
            _record.write_json(provenance_path, provenance)

            _log(log, f"command {shlex.join(command)}")
            output = UpstreamOutput()
            test_started = perf_counter()
            try:
                run_logged(
                    command,
                    log=log,
                    cwd=inputs.source,
                    env=build_test_environment(inputs.source),
                    on_line=output.consume,
                )
            finally:
                durations["test"] = perf_counter() - test_started
                output.close()
            _prepare.verify_source(inputs.source)

            if output.warning_count:
                write_status(
                    "warning",
                    "upstream emitted "
                    f"{output.warning_count} warning line(s); see run.log",
                )

            if not prediction_path.is_file():
                raise RuntimeError(
                    "MMDetection did not create predictions: "
                    f"{prediction_path}"
                )

            metrics, coco_durations = _evaluate(inputs, prediction_path)
            for phase, duration in coco_durations.items():
                durations[f"coco_{phase}"] = duration
            durations["coco_total"] = sum(coco_durations.values())

            expected = _record.load_expected(_EXPECTED)
            mismatches = _record.compare_metrics(metrics, expected)
            passed = not mismatches
            durations["total"] = perf_counter() - started

            summary = _record.success_summary(
                status="pass" if passed else "fail",
                started_at=started_at,
                metrics=metrics,
                expected=expected,
                mismatches=mismatches,
                durations=durations,
                prediction=prediction_path,
                warning_count=output.warning_count,
            )
            _record.write_json(summary_path, summary)
            _write_result(
                metrics,
                expected,
                mismatches,
                durations["total"],
            )
            return 0 if passed else 1
        except KeyboardInterrupt:
            durations["total"] = perf_counter() - started
            _record.write_json(
                summary_path,
                _record.failure_summary(
                    "interrupted",
                    started_at,
                    durations,
                    "KeyboardInterrupt",
                    "interrupted",
                ),
            )
            _log(log, "interrupted")
            raise
        except Exception as exc:
            durations["total"] = perf_counter() - started
            _record.write_json(
                summary_path,
                _record.failure_summary(
                    "error",
                    started_at,
                    durations,
                    type(exc).__name__,
                    str(exc),
                ),
            )
            traceback.print_exc(file=log)
            raise


def _parse_args(argv: list[str] | None) -> tuple[Path, Path, int]:
    parser = argparse.ArgumentParser(
        description="Reproduce RTMDet-tiny on COCO val2017."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache_dir(),
        help="input cache directory",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_ROOT / "runs" / "rtmdet",
        help="directory in which run directories are created",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="MMDetection test DataLoader workers (default: 2)",
    )
    args = parser.parse_args(argv)
    return (
        cast(Path, args.cache_dir),
        cast(Path, args.runs_dir),
        cast(int, args.num_workers),
    )


def default_cache_dir() -> Path:
    value = os.environ.get("XDG_CACHE_HOME")
    if value:
        root = Path(value).expanduser()
        if root.is_absolute():
            return root / "smallpotato"
    return Path.home() / ".cache" / "smallpotato"


def new_run_dir(root: Path, now: datetime, /) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    for suffix in range(100):
        name = run_id if suffix == 0 else f"{run_id}-{suffix}"
        path = root / name
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path

    raise RuntimeError(f"could not allocate run directory under {root}")


def _prepare_environment(log: TextIO) -> dict[str, str]:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not installed")

    env = _nested_environment()
    write_status("sync", "environment")
    run_logged(
        (uv, "sync", "--project", str(_ENV), "--locked"),
        log=log,
        cwd=_ROOT,
        env=env,
    )

    python = _ENV / ".venv" / "bin" / "python"
    run_logged(
        (uv, "pip", "check", "--python", str(python)),
        log=log,
        cwd=_ROOT,
        env=env,
    )

    capture = SmokeCapture()
    env = env.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    run_logged(
        (str(python), str(_SMOKE)),
        log=log,
        cwd=_ROOT,
        env=env,
        on_line=capture.consume,
    )
    values = capture.result()
    values["uv"] = capture_output((uv, "--version"))

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        values["nvidia_smi"] = capture_output(
            (
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            )
        )
    return values


def _test_command(
    inputs: _prepare.PreparedInputs,
    runtime: Path,
    upstream: Path,
    /,
) -> tuple[str, ...]:
    python = _ENV / ".venv" / "bin" / "python"
    return (
        str(python),
        str(inputs.source / "tools" / "test.py"),
        str(runtime),
        str(inputs.checkpoint.path),
        "--work-dir",
        str(upstream),
        "--launcher",
        "none",
    )


def build_test_environment(source: Path, /) -> dict[str, str]:
    env = _nested_environment()
    env["PYTHONPATH"] = str(source)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _nested_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env


def _evaluate(
    inputs: _prepare.PreparedInputs,
    prediction: Path,
    /,
) -> tuple[COCOMetrics, dict[str, float]]:
    timer = _PhaseTimer()
    metrics = eval_coco(
        inputs.annotations,
        prediction,
        on_phase=timer.begin,
    )
    return metrics, timer.finish()


def _write_header(run: Path) -> None:
    print("smallpotato - reproduce rtmdet", file=sys.stderr)
    print(file=sys.stderr)
    write_status("model", "rtmdet-tiny")
    write_status("dataset", "coco/val2017")
    write_status("run", str(run))
    print(file=sys.stderr)


def _write_input_checks() -> None:
    _write_check("source")
    _write_check("data")
    _write_check("checkpoint")


def _write_check(name: str, /) -> None:
    write_status("check", f"{name:<28}ok")


def _write_result(
    metrics: COCOMetrics,
    expected: _record.ExpectedResult,
    mismatches: list[_record.MetricMismatch],
    total: float,
    /,
) -> None:
    print(file=sys.stderr)
    _write_metrics(metrics)
    print(file=sys.stderr)
    write_status("expected", f"AP {expected.metrics.ap:.4f}")
    write_status("observed", f"AP {metrics.ap:.4f}")
    write_status("tolerance", f"absolute {expected.tolerance:.4f}")
    if mismatches:
        first = mismatches[0]
        write_status("mismatch", f"{first.metric} {first.error:.4f}")
    write_status("result", "pass" if not mismatches else "fail")
    print(file=sys.stderr)
    write_status("write", "summary.json")
    write_status("write", "provenance.json")
    write_status("write", "pred.bbox.json")
    write_status("log", "run.log")
    print(file=sys.stderr)
    write_status("done", format_duration(total))


def _write_metrics(metrics: COCOMetrics, /) -> None:
    write_status(
        "metric",
        "   AP      AP50      AP75       APs       APm       APl",
    )
    write_status(
        "",
        f"{metrics.ap:7.4f}   {metrics.ap50:7.4f}   "
        f"{metrics.ap75:7.4f}   {metrics.ap_small:7.4f}   "
        f"{metrics.ap_medium:7.4f}   {metrics.ap_large:7.4f}",
    )
    write_status(
        "metric",
        "  AR1      AR10     AR100       ARs       ARm       ARl",
    )
    write_status(
        "",
        f"{metrics.ar1:7.4f}   {metrics.ar10:7.4f}   "
        f"{metrics.ar100:7.4f}   {metrics.ar_small:7.4f}   "
        f"{metrics.ar_medium:7.4f}   {metrics.ar_large:7.4f}",
    )


def _log_inputs(log: TextIO, inputs: _prepare.PreparedInputs, /) -> None:
    _log(log, f"source {inputs.source}")
    source_transport = (
        _prepare.MMDET_REPOSITORY if inputs.source_fetched else "cache"
    )
    _log(log, f"source transport {source_transport}")
    _log(log, f"checkpoint {inputs.checkpoint.path}")
    _log(log, f"checkpoint transport {_transport(inputs.checkpoint)}")
    _log(log, f"coco {inputs.coco}")
    _log(log, f"val2017 transport {_transport(inputs.val_archive)}")
    _log(
        log,
        f"annotations transport {_transport(inputs.annotations_archive)}",
    )


def _transport(file: _prepare.AcquiredFile, /) -> str:
    return "cache" if file.source is None else file.source


def _log(log: TextIO, message: str, /) -> None:
    log.write(f"[smallpotato] {message}\n")
    log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
