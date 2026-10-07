__all__ = ()

import json
import os
import platform
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from typing import cast, final

from reproductions.rtmdet import _prepare

from smallpotato._download import sha256_file
from smallpotato._process import capture_output
from smallpotato.evaluation import COCOMetrics

_METRIC_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar100",
    "ar_small",
    "ar_medium",
    "ar_large",
)


@final
@dataclass(frozen=True, slots=True, match_args=False)
class ExpectedResult:
    metrics: COCOMetrics
    tolerance: float


@final
@dataclass(frozen=True, slots=True, match_args=False)
class MetricMismatch:
    metric: str
    expected: float
    observed: float

    @property
    def error(self) -> float:
        return abs(self.observed - self.expected)


def load_expected(path: Path, /) -> ExpectedResult:
    raw_document: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_document, dict):
        raise RuntimeError(f"invalid expected result: {path}")
    document = cast(dict[str, object], raw_document)

    tolerance = document.get("absolute_tolerance")
    raw_metrics = document.get("metrics")
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise RuntimeError(f"invalid absolute_tolerance in {path}")
    if not isinstance(raw_metrics, dict):
        raise RuntimeError(f"invalid metrics in {path}")
    metric_values = cast(dict[str, object], raw_metrics)

    metrics = COCOMetrics(
        ap=_metric(metric_values, "ap", path),
        ap50=_metric(metric_values, "ap50", path),
        ap75=_metric(metric_values, "ap75", path),
        ap_small=_metric(metric_values, "ap_small", path),
        ap_medium=_metric(metric_values, "ap_medium", path),
        ap_large=_metric(metric_values, "ap_large", path),
        ar1=_metric(metric_values, "ar1", path),
        ar10=_metric(metric_values, "ar10", path),
        ar100=_metric(metric_values, "ar100", path),
        ar_small=_metric(metric_values, "ar_small", path),
        ar_medium=_metric(metric_values, "ar_medium", path),
        ar_large=_metric(metric_values, "ar_large", path),
    )
    if set(metric_values) != set(_METRIC_NAMES):
        raise RuntimeError(f"unexpected metric names in {path}")

    return ExpectedResult(metrics=metrics, tolerance=float(tolerance))


def compare_metrics(
    observed: COCOMetrics,
    expected: ExpectedResult,
    /,
) -> list[MetricMismatch]:
    mismatches: list[MetricMismatch] = []
    observed_values = _metric_values(observed)
    expected_values = _metric_values(expected.metrics)
    for name in _METRIC_NAMES:
        actual = observed_values[name]
        target = expected_values[name]
        error = abs(actual - target)
        if error > expected.tolerance:
            mismatches.append(
                MetricMismatch(
                    metric=name,
                    expected=target,
                    observed=actual,
                )
            )
    return mismatches


def build_provenance(
    *,
    root: Path,
    expected: Path,
    environment_lock: Path,
    started_at: datetime,
    cache: Path,
    environment: dict[str, str],
    command: tuple[str, ...],
    num_workers: int,
    runtime: Path,
    inputs: _prepare.PreparedInputs,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "started_at": iso_utc(started_at),
        "smallpotato": {
            "version": version("smallpotato"),
            "pycocotools": version("pycocotools"),
            "lock_sha256": sha256_file(root / "uv.lock"),
            "git": _git_state(root),
        },
        "reproduction": {
            "model": "rtmdet-tiny",
            "dataset": "coco/val2017",
            "upstream_repository": _prepare.MMDET_REPOSITORY,
            "upstream_revision": _prepare.MMDET_REVISION,
            "upstream_config": str(_prepare.MMDET_CONFIG),
            "expected_sha256": sha256_file(expected),
        },
        "environment": {
            **environment,
            "lock_sha256": sha256_file(environment_lock),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "root_python": platform.python_version(),
        },
        "inputs": {
            "cache": str(cache.expanduser().resolve()),
            "upstream_source": {
                "path": str(inputs.source),
                "cache_hit": not inputs.source_fetched,
            },
            "checkpoint": _acquired_artifact(
                _prepare.CHECKPOINT, inputs.checkpoint
            ),
            "coco_val2017": _acquired_artifact(
                _prepare.VAL2017, inputs.val_archive
            ),
            "coco_annotations": _acquired_artifact(
                _prepare.ANNOTATIONS, inputs.annotations_archive
            ),
            "instances_val2017": {
                "path": str(inputs.annotations),
                "sha256": _prepare.INSTANCES_VAL_SHA256,
                "size": _prepare.INSTANCES_VAL_SIZE,
            },
        },
        "runtime": {
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "matplotlib_backend": "Agg",
            "python_unbuffered": True,
            "python_dont_write_bytecode": True,
            "format_only": True,
            "runtime_config": {
                "path": str(runtime),
                "sha256": sha256_file(runtime),
            },
            "command": list(command),
        },
    }


def success_summary(
    *,
    status: str,
    started_at: datetime,
    metrics: COCOMetrics,
    expected: ExpectedResult,
    mismatches: list[MetricMismatch],
    durations: dict[str, float],
    prediction: Path,
    warning_count: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "started_at": iso_utc(started_at),
        "finished_at": iso_utc(datetime.now(UTC)),
        "metrics": _metric_values(metrics),
        "expected": {
            "absolute_tolerance": expected.tolerance,
            "metrics": _metric_values(expected.metrics),
        },
        "mismatches": [_mismatch_values(item) for item in mismatches],
        "durations_seconds": durations,
        "upstream_warning_lines": warning_count,
        "artifacts": {
            "predictions": {
                "path": prediction.name,
                "size": prediction.stat().st_size,
                "sha256": sha256_file(prediction),
            },
            "runtime_config": "runtime.py",
            "upstream": "upstream",
            "log": "run.log",
            "provenance": "provenance.json",
        },
    }


def failure_summary(
    status: str,
    started_at: datetime,
    durations: dict[str, float],
    error_type: str,
    message: str,
    /,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "started_at": iso_utc(started_at),
        "finished_at": iso_utc(datetime.now(UTC)),
        "durations_seconds": durations,
        "error": {"type": error_type, "message": message},
    }


def write_json(path: Path, document: object, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(
                document,
                file,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def iso_utc(value: datetime, /) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _metric(metrics: dict[str, object], name: str, path: Path) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"invalid metric {name!r} in {path}")
    return float(value)


def _acquired_artifact(
    artifact: _prepare.Artifact,
    acquired: _prepare.AcquiredFile,
    /,
) -> dict[str, object]:
    return {
        "path": str(acquired.path),
        "filename": artifact.filename,
        "sha256": artifact.sha256,
        "size": artifact.size,
        "allowed_sources": list(artifact.urls),
        "transport": {
            "cache_hit": acquired.source is None,
            "source": acquired.source,
        },
    }


def _mismatch_values(item: MetricMismatch) -> dict[str, object]:
    return {
        "metric": item.metric,
        "expected": item.expected,
        "observed": item.observed,
        "error": item.error,
    }


def _metric_values(metrics: COCOMetrics) -> dict[str, float]:
    return {
        "ap": metrics.ap,
        "ap50": metrics.ap50,
        "ap75": metrics.ap75,
        "ap_small": metrics.ap_small,
        "ap_medium": metrics.ap_medium,
        "ap_large": metrics.ap_large,
        "ar1": metrics.ar1,
        "ar10": metrics.ar10,
        "ar100": metrics.ar100,
        "ar_small": metrics.ar_small,
        "ar_medium": metrics.ar_medium,
        "ar_large": metrics.ar_large,
    }


def _git_state(root: Path, /) -> dict[str, object]:
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        return {"commit": None, "dirty": None}

    try:
        commit = capture_output(
            (git, "-C", str(root), "rev-parse", "--verify", "HEAD")
        )
    except CalledProcessError:
        commit = ""
    status = capture_output((git, "-C", str(root), "status", "--porcelain"))
    return {"commit": commit or None, "dirty": bool(status)}
