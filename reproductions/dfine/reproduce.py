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
from typing import TextIO, cast

from reproductions.dfine import _prepare, _record

from smallpotato._console import Progress, format_duration, write_status
from smallpotato._process import run_captured, run_logged
from smallpotato._record import (
    iso_utc,
    load_reference,
    metric_errors,
    write_json,
)
from smallpotato.evaluation import COCOMetrics, eval_coco

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_ENV = _HERE / "env"
_PYTHON = _ENV / ".venv" / "bin" / "python"
_INFER = _HERE / "infer.py"
_REFERENCE = _HERE / "reference.json"
_SMOKE = _HERE / "smoke.py"

_SMOKE_LABEL_WIDTH = 14
_TEST_PROGRESS = re.compile(
    r"\bTest:\s+\[\s*(\d+)/(\d+)\]"
    r"(?:\s+eta:\s*(\d+):(\d+):(\d+))?"
)
_WARNING = re.compile(r"\bWARNING\b|\b[A-Za-z]*Warning:")
_SMOKE_KEYS = (
    ("python", "python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torch cuda", "torch_cuda"),
    ("torchvision", "torchvision"),
    ("device", "device"),
    ("state tensors", "state_tensors"),
    ("checkpoint", "checkpoint"),
    ("dfine model", "dfine_model"),
)


class _UpstreamOutput:
    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._last_progress = 0
        self.warning_count = 0

    def consume(self, line: str, /) -> None:
        match = _TEST_PROGRESS.search(line)
        if match is not None:
            current = int(match.group(1)) + 1
            total = int(match.group(2))
            if current > self._last_progress:
                if self._progress is None:
                    self._progress = Progress("test", total)
                self._progress.update(
                    current,
                    remaining=_remaining_seconds(match),
                )
                self._last_progress = current

        if _WARNING.search(line) is not None:
            self.warning_count += 1

    def close(self) -> None:
        if self._progress is not None:
            self._progress.close()


class _SmokeCapture:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def consume(self, line: str, /) -> None:
        for label, key in _SMOKE_KEYS:
            prefix = f"{label:<{_SMOKE_LABEL_WIDTH}}"
            if line.startswith(prefix):
                self._values[key] = line[len(prefix) :].strip()
                return

    def result(self) -> dict[str, str]:
        expected = {key for _, key in _SMOKE_KEYS}
        missing = expected - self._values.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise RuntimeError(f"incomplete environment smoke output: {names}")
        return dict(self._values)


def main(argv: list[str] | None = None) -> int:
    cache, runs, batch_size, num_workers = _parse_args(argv)
    try:
        return reproduce(
            cache=cache,
            runs=runs,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    except KeyboardInterrupt:
        return 130


def reproduce(
    *,
    cache: Path,
    runs: Path,
    batch_size: int,
    num_workers: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    started_at = datetime.now(UTC)
    started = perf_counter()
    run = _new_run_dir(runs, started_at)
    log_path = run / "run.log"
    summary_path = run / "summary.json"
    provenance_path = run / "provenance.json"
    prediction_path = run / "pred.bbox.json"
    durations: dict[str, float] = {}

    _write_header(run)

    with log_path.open(
        "w", encoding="utf-8", newline="\n", buffering=1
    ) as log:
        _log(log, f"started {iso_utc(started_at)}")
        try:
            write_status("prepare", "inputs")
            phase = perf_counter()
            inputs = _prepare.prepare_inputs(cache)
            durations["inputs"] = perf_counter() - phase
            _log(log, f"source {inputs.source}")
            _log(log, f"checkpoint {inputs.checkpoint}")
            _log(log, f"coco {inputs.coco}")
            _write_check("source")
            _write_check("data")
            _write_check("checkpoint")

            reference = load_reference(_REFERENCE)

            phase = perf_counter()
            environment = _prepare_environment(log, inputs)
            durations["environment"] = perf_counter() - phase
            write_status("device", environment["device"])
            _write_check("environment")
            _prepare.verify_source(inputs.source)

            command = _infer_command(
                inputs,
                prediction_path,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            write_json(
                provenance_path,
                _record.build_provenance(
                    root=_ROOT,
                    reference=_REFERENCE,
                    environment_lock=_ENV / "uv.lock",
                    started_at=started_at,
                    environment=environment,
                    command=command,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ),
            )
            _log(log, f"command {shlex.join(command)}")

            output = _UpstreamOutput()
            phase = perf_counter()
            try:
                run_logged(
                    command,
                    log=log,
                    cwd=inputs.source,
                    env=_upstream_environment(inputs.source),
                    on_line=output.consume,
                )
            finally:
                durations["inference"] = perf_counter() - phase
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
                    f"D-FINE did not create predictions: {prediction_path}"
                )

            phase = perf_counter()
            metrics = eval_coco(
                inputs.annotations,
                prediction_path,
                on_phase=_coco_phase,
            )
            durations["evaluation"] = perf_counter() - phase
            durations["total"] = perf_counter() - started

            errors = metric_errors(metrics, reference)
            write_json(
                summary_path,
                _record.success_summary(
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    metrics=metrics,
                    reference=reference,
                    durations=durations,
                    prediction=prediction_path,
                    warning_count=output.warning_count,
                ),
            )
            _write_result(
                metrics,
                reference,
                max(errors.values()),
                durations["total"],
            )
            return 0
        except KeyboardInterrupt:
            _log(log, "interrupted")
            raise
        except Exception:
            traceback.print_exc(file=log)
            log.flush()
            raise


def _parse_args(argv: list[str] | None) -> tuple[Path, Path, int, int]:
    parser = argparse.ArgumentParser(
        description="Reproduce D-FINE-N on COCO val2017."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="input cache directory",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=_ROOT / "runs" / "dfine",
        help="directory in which run directories are created",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="single-GPU validation batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="D-FINE validation DataLoader workers (default: 4)",
    )
    args = parser.parse_args(argv)
    return (
        cast(Path, args.cache_dir),
        cast(Path, args.runs_dir),
        cast(int, args.batch_size),
        cast(int, args.num_workers),
    )


def _default_cache_dir() -> Path:
    value = os.environ.get("XDG_CACHE_HOME")
    if value:
        root = Path(value).expanduser()
        if root.is_absolute():
            return root / "smallpotato"
    return Path.home() / ".cache" / "smallpotato"


def _new_run_dir(root: Path, now: datetime, /) -> Path:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = now.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = 0

    while True:
        name = run_id if suffix == 0 else f"{run_id}-{suffix}"
        path = root / name
        try:
            path.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return path


def _prepare_environment(
    log: TextIO,
    inputs: _prepare.PreparedInputs,
) -> dict[str, str]:
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
    run_logged(
        (uv, "pip", "check", "--python", str(_PYTHON)),
        log=log,
        cwd=_ROOT,
        env=env,
    )

    capture = _SmokeCapture()
    run_logged(
        (
            str(_PYTHON),
            str(_SMOKE),
            str(inputs.config),
            str(inputs.checkpoint),
        ),
        log=log,
        cwd=inputs.source,
        env=_upstream_environment(inputs.source),
        on_line=capture.consume,
    )

    values = capture.result()
    values["uv"] = run_captured((uv, "--version"), env=env).strip()

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        values["nvidia_smi"] = run_captured(
            (
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ),
            env=env,
        ).strip()
    return values


def _infer_command(
    inputs: _prepare.PreparedInputs,
    output: Path,
    /,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[str, ...]:
    return (
        str(_PYTHON),
        str(_INFER),
        "--config",
        str(inputs.config),
        "--checkpoint",
        str(inputs.checkpoint),
        "--images",
        str(inputs.images),
        "--annotations",
        str(inputs.annotations),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
    )


def _upstream_environment(source: Path, /) -> dict[str, str]:
    env = _nested_environment()
    env["PYTHONPATH"] = str(source)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _nested_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "VIRTUAL_ENV",
        "PYTHONHOME",
        "PYTHONPATH",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(name, None)
    return env


def _remaining_seconds(match: re.Match[str], /) -> int | None:
    hours, minutes, seconds = match.group(3, 4, 5)
    if hours is None or minutes is None or seconds is None:
        return None
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _coco_phase(phase: str, /) -> None:
    write_status("coco", phase)


def _write_header(run: Path, /) -> None:
    print("smallpotato - reproduce dfine", file=sys.stderr)
    print(file=sys.stderr)
    write_status("model", "dfine-n")
    write_status("dataset", "coco/val2017")
    write_status("run", str(run))
    print(file=sys.stderr)


def _write_check(name: str, /) -> None:
    write_status("check", f"{name:<28}ok")


def _write_result(
    metrics: COCOMetrics,
    reference: COCOMetrics,
    max_error: float,
    total: float,
    /,
) -> None:
    print(file=sys.stderr)
    _write_metrics(metrics)
    print(file=sys.stderr)
    write_status("reference", f"AP {reference.ap:.4f}")
    write_status("observed", f"AP {metrics.ap:.4f}")
    write_status("difference", f"max absolute {max_error:.6f}")
    print(file=sys.stderr)
    write_status("write", "summary.json")
    write_status("write", "provenance.json")
    write_status("write", "pred.bbox.json")
    write_status("log", "run.log")
    print(file=sys.stderr)
    write_status("done", format_duration(total))


def _write_metrics(metrics: COCOMetrics, /) -> None:
    write_status(
        "metric", "   AP      AP50      AP75       APs       APm       APl"
    )
    write_status(
        "",
        f"{metrics.ap:7.4f}   {metrics.ap50:7.4f}   "
        f"{metrics.ap75:7.4f}   {metrics.ap_small:7.4f}   "
        f"{metrics.ap_medium:7.4f}   {metrics.ap_large:7.4f}",
    )
    write_status(
        "metric", "  AR1      AR10     AR100       ARs       ARm       ARl"
    )
    write_status(
        "",
        f"{metrics.ar1:7.4f}   {metrics.ar10:7.4f}   "
        f"{metrics.ar100:7.4f}   {metrics.ar_small:7.4f}   "
        f"{metrics.ar_medium:7.4f}   {metrics.ar_large:7.4f}",
    )


def _log(log: TextIO, message: str, /) -> None:
    log.write(f"[smallpotato] {message}\n")
    log.flush()


if __name__ == "__main__":
    raise SystemExit(main())
