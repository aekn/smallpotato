import platform
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from reproductions.dfine import _prepare

from smallpotato import _coco
from smallpotato._download import sha256_file
from smallpotato._record import (
    artifact_identity,
    git_state,
    iso_utc,
    metric_errors,
    metric_values,
)
from smallpotato.evaluation import COCOMetrics


def build_provenance(
    *,
    root: Path,
    reference: Path,
    environment_lock: Path,
    started_at: datetime,
    environment: Mapping[str, str],
    command: Sequence[str],
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    commit, dirty = git_state(root)
    return {
        "started_at": iso_utc(started_at),
        "smallpotato": {
            "commit": commit,
            "dirty": dirty,
            "lock_sha256": sha256_file(root / "uv.lock"),
            "pycocotools": version("pycocotools"),
        },
        "upstream": {
            "repository": _prepare.DFINE_REPOSITORY,
            "revision": _prepare.DFINE_REVISION,
            "config": str(_prepare.DFINE_CONFIG),
        },
        "environment": {
            **dict(environment),
            "lock_sha256": sha256_file(environment_lock),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "root_python": platform.python_version(),
        },
        "inputs": {
            "checkpoint": artifact_identity(_prepare.CHECKPOINT),
            "coco_val2017": artifact_identity(_coco.VAL2017),
            "coco_annotations": artifact_identity(_coco.ANNOTATIONS),
            "instances_val2017": {
                "sha256": _coco.INSTANCES_VAL2017_SHA256,
                "size": _coco.INSTANCES_VAL2017_SIZE,
            },
        },
        "runtime": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "reference_world_size": 4,
            "reference_total_batch_size": 256,
            "reference_local_batch_size": 64,
            "reference_sha256": sha256_file(reference),
            "command": list(command),
        },
    }


def success_summary(
    *,
    started_at: datetime,
    finished_at: datetime,
    metrics: COCOMetrics,
    reference: COCOMetrics,
    durations: Mapping[str, float],
    prediction: Path,
    warning_count: int,
) -> dict[str, object]:
    errors = metric_errors(metrics, reference)
    return {
        "status": "complete",
        "started_at": iso_utc(started_at),
        "finished_at": iso_utc(finished_at),
        "metrics": metric_values(metrics),
        "reference": {
            "metrics": metric_values(reference),
            "max_abs_error": max(errors.values()),
        },
        "durations_seconds": dict(durations),
        "upstream_warning_lines": warning_count,
        "prediction": {
            "path": prediction.name,
            "size": prediction.stat().st_size,
            "sha256": sha256_file(prediction),
        },
    }
