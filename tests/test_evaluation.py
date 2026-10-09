import json
from pathlib import Path

import pytest

from smallpotato.evaluation import eval_coco


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_annotations(path: Path) -> None:
    _write_json(
        path,
        {
            "images": [
                {"id": 1, "width": 256, "height": 256},
                {"id": 2, "width": 256, "height": 256},
                {"id": 3, "width": 512, "height": 512},
            ],
            "categories": [{"id": 1, "name": "object"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [20, 20, 50, 50],
                    "area": 2500,
                    "iscrowd": 0,
                },
                {
                    "id": 3,
                    "image_id": 3,
                    "category_id": 1,
                    "bbox": [40, 40, 150, 150],
                    "area": 22500,
                    "iscrowd": 0,
                },
            ],
        },
    )


def _write_perfect_predictions(path: Path) -> None:
    _write_json(
        path,
        [
            {
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "score": 1.0,
            },
            {
                "image_id": 2,
                "category_id": 1,
                "bbox": [20, 20, 50, 50],
                "score": 1.0,
            },
            {
                "image_id": 3,
                "category_id": 1,
                "bbox": [40, 40, 150, 150],
                "score": 1.0,
            },
        ],
    )


def test_eval_coco_perfect_predictions(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    predictions = tmp_path / "predictions.json"
    _write_annotations(annotations)
    _write_perfect_predictions(predictions)

    phases: list[str] = []
    metrics = eval_coco(
        annotations,
        predictions,
        on_phase=phases.append,
    )

    assert phases == ["load", "evaluate", "accumulate", "summarize"]
    assert metrics.ap == pytest.approx(1.0)
    assert metrics.ap50 == pytest.approx(1.0)
    assert metrics.ap75 == pytest.approx(1.0)
    assert metrics.ap_small == pytest.approx(1.0)
    assert metrics.ap_medium == pytest.approx(1.0)
    assert metrics.ap_large == pytest.approx(1.0)
    assert metrics.ar100 == pytest.approx(1.0)


def test_eval_coco_empty_predictions(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    predictions = tmp_path / "predictions.json"
    _write_annotations(annotations)
    _write_json(predictions, [])

    metrics = eval_coco(annotations, predictions)

    assert metrics.ap == pytest.approx(0.0)
    assert metrics.ar100 == pytest.approx(0.0)


def test_eval_coco_requires_prediction_array(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    predictions = tmp_path / "predictions.json"
    _write_json(annotations, {})
    _write_json(predictions, {})

    with pytest.raises(TypeError, match="JSON array"):
        eval_coco(annotations, predictions)
