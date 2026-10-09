import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CalledProcessError
from typing import cast

from ._download import Artifact
from ._process import run_captured
from .evaluation import COCOMetrics

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


def load_reference(path: Path, /) -> COCOMetrics:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid reference result: {path}")
    document = cast(dict[str, object], raw)

    raw_metrics = document.get("metrics")
    if not isinstance(raw_metrics, dict):
        raise RuntimeError(f"invalid metrics in {path}")
    metrics = cast(dict[str, object], raw_metrics)

    if set(metrics) != set(_METRIC_NAMES):
        raise RuntimeError(f"invalid metrics in {path}")

    return COCOMetrics(
        ap=_metric(metrics, "ap", path),
        ap50=_metric(metrics, "ap50", path),
        ap75=_metric(metrics, "ap75", path),
        ap_small=_metric(metrics, "ap_small", path),
        ap_medium=_metric(metrics, "ap_medium", path),
        ap_large=_metric(metrics, "ap_large", path),
        ar1=_metric(metrics, "ar1", path),
        ar10=_metric(metrics, "ar10", path),
        ar100=_metric(metrics, "ar100", path),
        ar_small=_metric(metrics, "ar_small", path),
        ar_medium=_metric(metrics, "ar_medium", path),
        ar_large=_metric(metrics, "ar_large", path),
    )


def metric_values(metrics: COCOMetrics, /) -> dict[str, float]:
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


def metric_errors(
    observed: COCOMetrics,
    reference: COCOMetrics,
    /,
) -> dict[str, float]:
    actual = metric_values(observed)
    expected = metric_values(reference)
    return {name: abs(actual[name] - expected[name]) for name in _METRIC_NAMES}


def artifact_identity(artifact: Artifact, /) -> dict[str, object]:
    identity: dict[str, object] = {
        "filename": artifact.filename,
        "sha256": artifact.sha256,
    }
    if artifact.size is not None:
        identity["size"] = artifact.size
    return identity


def git_state(root: Path, /) -> tuple[str | None, bool | None]:
    git = shutil.which("git")
    if git is None or not (root / ".git").exists():
        return None, None

    try:
        commit = run_captured(
            (git, "-C", str(root), "rev-parse", "--verify", "HEAD")
        ).strip()
        status = run_captured(
            (
                git,
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            )
        )
    except CalledProcessError:
        return None, None

    return commit or None, bool(status)


def write_json(path: Path, document: object, /) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def iso_utc(value: datetime, /) -> str:
    if value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _metric(values: dict[str, object], name: str, path: Path, /) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"invalid metric {name!r} in {path}")

    metric = float(value)
    if not math.isfinite(metric):
        raise RuntimeError(f"non-finite metric {name!r} in {path}")
    return metric
