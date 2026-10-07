import json
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from smallpotato.evaluation import eval_coco


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_ann(path: Path) -> None:
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


def _write_perfect_pred(path: Path) -> None:
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


def test_eval_coco_perfect_pred(tmp_path: Path) -> None:
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"

    _write_ann(ann)
    _write_perfect_pred(pred)

    metrics = eval_coco(ann, pred)

    assert metrics.ap == pytest.approx(1.0)
    assert metrics.ap50 == pytest.approx(1.0)
    assert metrics.ap75 == pytest.approx(1.0)
    assert metrics.ap_small == pytest.approx(1.0)
    assert metrics.ap_medium == pytest.approx(1.0)
    assert metrics.ap_large == pytest.approx(1.0)
    assert metrics.ar100 == pytest.approx(1.0)


def test_eval_coco_empty_pred(tmp_path: Path) -> None:
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"

    _write_ann(ann)
    _write_json(pred, [])

    metrics = eval_coco(ann, pred)

    assert metrics.ap == pytest.approx(0.0)
    assert metrics.ar100 == pytest.approx(0.0)


def test_eval_coco_requires_array(tmp_path: Path) -> None:
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"

    _write_json(ann, {})
    _write_json(pred, {})

    with pytest.raises(TypeError, match="JSON array"):
        eval_coco(ann, pred)


def test_eval_coco_passes_loaded_predictions_to_coco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"

    _write_ann(ann)
    _write_perfect_pred(pred)

    original_load_res = COCO.loadRes
    seen: list[object] = []

    def load_res(self: COCO, res_file: object) -> COCO:
        seen.append(res_file)
        return original_load_res(
            self,
            res_file,  # pyright: ignore[reportArgumentType]
        )

    monkeypatch.setattr(COCO, "loadRes", load_res)

    eval_coco(ann, pred)

    assert len(seen) == 1
    assert isinstance(seen[0], list)


def test_eval_coco_reports_phases(tmp_path: Path) -> None:
    ann = tmp_path / "ann.json"
    pred = tmp_path / "pred.json"

    _write_ann(ann)
    _write_perfect_pred(pred)
    phases: list[str] = []

    eval_coco(ann, pred, on_phase=phases.append)

    assert phases == ["load", "evaluate", "accumulate", "summarize"]
