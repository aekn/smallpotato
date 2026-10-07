import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from reproductions.rtmdet import _prepare, _record
from smallpotato.evaluation import COCOMetrics


def _metrics(*, ap: float = 0.4112) -> COCOMetrics:
    return COCOMetrics(
        ap=ap,
        ap50=0.5787,
        ap75=0.4472,
        ap_small=0.2099,
        ap_medium=0.4551,
        ap_large=0.5830,
        ar1=0.3340,
        ar10=0.5536,
        ar100=0.6053,
        ar_small=0.3807,
        ar_medium=0.6769,
        ar_large=0.7969,
    )


def _write_expected(path: Path, metrics: COCOMetrics) -> None:
    path.write_text(
        json.dumps(
            {
                "absolute_tolerance": 0.002,
                "metrics": {
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
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_expected_result(tmp_path: Path) -> None:
    path = tmp_path / "expected.json"
    metrics = _metrics()
    _write_expected(path, metrics)

    expected = _record.load_expected(path)

    assert expected.metrics == metrics
    assert expected.tolerance == pytest.approx(0.002)


def test_load_expected_rejects_unknown_metric(tmp_path: Path) -> None:
    path = tmp_path / "expected.json"
    _write_expected(path, _metrics())
    document = cast(
        dict[str, object],
        json.loads(path.read_text(encoding="utf-8")),
    )
    metrics = cast(dict[str, object], document["metrics"])
    metrics["other"] = 1.0
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected metric names"):
        _record.load_expected(path)


def test_compare_metrics_reports_only_out_of_tolerance() -> None:
    expected = _record.ExpectedResult(metrics=_metrics(), tolerance=0.002)
    observed = _metrics(ap=0.408)

    mismatches = _record.compare_metrics(observed, expected)

    assert len(mismatches) == 1
    mismatch = mismatches[0]
    assert mismatch.metric == "ap"
    assert mismatch.expected == pytest.approx(0.4112)
    assert mismatch.observed == pytest.approx(0.408)
    assert mismatch.error == pytest.approx(0.0032)


def test_build_provenance_records_transport_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "uv.lock").write_text("root lock\n", encoding="utf-8")
    expected = tmp_path / "expected.json"
    expected.write_text("{}\n", encoding="utf-8")
    environment_lock = tmp_path / "environment.lock"
    environment_lock.write_text("environment lock\n", encoding="utf-8")
    runtime = tmp_path / "runtime.py"
    runtime.write_text("test = True\n", encoding="utf-8")

    checkpoint = _prepare.AcquiredFile(
        path=tmp_path / "checkpoint.pth",
        source="https://example.test/checkpoint.pth",
    )
    cached_val = _prepare.AcquiredFile(
        path=tmp_path / "val2017.zip",
        source=None,
    )
    cached_annotations = _prepare.AcquiredFile(
        path=tmp_path / "annotations.zip",
        source=None,
    )
    inputs = _prepare.PreparedInputs(
        source=tmp_path / "mmdetection",
        source_fetched=True,
        config=tmp_path / "config.py",
        checkpoint=checkpoint,
        coco=tmp_path / "coco",
        annotations=tmp_path / "instances_val2017.json",
        val_archive=cached_val,
        annotations_archive=cached_annotations,
    )

    def version(_: str) -> str:
        return "test-version"

    monkeypatch.setattr(_record, "version", version)

    provenance = _record.build_provenance(
        root=root,
        expected=expected,
        environment_lock=environment_lock,
        started_at=datetime(2026, 10, 6, tzinfo=UTC),
        cache=tmp_path / "cache",
        environment={"device": "Tesla T4"},
        command=("python", "test.py"),
        num_workers=2,
        runtime=runtime,
        inputs=inputs,
    )

    raw_inputs = cast(dict[str, object], provenance["inputs"])
    raw_checkpoint = cast(dict[str, object], raw_inputs["checkpoint"])
    transport = cast(dict[str, object], raw_checkpoint["transport"])
    assert transport == {
        "cache_hit": False,
        "source": "https://example.test/checkpoint.pth",
    }
    raw_val = cast(dict[str, object], raw_inputs["coco_val2017"])
    assert cast(dict[str, object], raw_val["transport"])["cache_hit"] is True


def test_write_json_replaces_document(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"

    _record.write_json(path, {"status": "first"})
    _record.write_json(path, {"status": "pass"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "pass"}
    assert list(tmp_path.iterdir()) == [path]


def test_checked_in_expected_result_is_valid() -> None:
    path = Path("reproductions/rtmdet/expected.json")

    expected = _record.load_expected(path)

    assert expected.tolerance >= 0
