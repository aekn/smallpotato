__all__ = ("COCOMetrics", "eval_coco")

import io
import json
from collections.abc import Callable
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import final

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@final
@dataclass(frozen=True, slots=True, match_args=False)
class COCOMetrics:
    """COCO bounding-box evaluation metrics."""

    ap: float
    ap50: float
    ap75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar1: float
    ar10: float
    ar100: float
    ar_small: float
    ar_medium: float
    ar_large: float


def eval_coco(
    ann: Path,
    pred: Path,
    /,
    *,
    on_phase: Callable[[str], None] | None = None,
) -> COCOMetrics:
    """Evaluate COCO bounding-box predictions."""

    notify = on_phase or _ignore_phase
    notify("load")

    with pred.open(encoding="utf-8") as file:
        doc: object = json.load(file)

    if not isinstance(doc, list):
        raise TypeError("predictions must contain a JSON array")

    with redirect_stdout(io.StringIO()):
        gt = COCO(ann)
        if doc:
            dt = gt.loadRes(doc)  # pyright: ignore[reportArgumentType]
        else:
            dt = _empty_dt(gt)

        evaluator = COCOeval(gt, dt, "bbox")

        notify("evaluate")
        evaluator.evaluate()
        notify("accumulate")
        evaluator.accumulate()
        notify("summarize")
        evaluator.summarize()

    stats = evaluator.stats

    return COCOMetrics(
        ap=float(stats[0]),
        ap50=float(stats[1]),
        ap75=float(stats[2]),
        ap_small=float(stats[3]),
        ap_medium=float(stats[4]),
        ap_large=float(stats[5]),
        ar1=float(stats[6]),
        ar10=float(stats[7]),
        ar100=float(stats[8]),
        ar_small=float(stats[9]),
        ar_medium=float(stats[10]),
        ar_large=float(stats[11]),
    )


def _ignore_phase(_phase: str, /) -> None:
    pass


def _empty_dt(gt: COCO, /) -> COCO:
    dt = COCO()
    dt.dataset = deepcopy(gt.dataset)
    dt.dataset["annotations"] = []
    dt.createIndex()
    return dt
