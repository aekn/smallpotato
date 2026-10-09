import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

import torch
from src.core import YAMLConfig
from src.data.dataset import mscoco_category2name
from src.misc import MetricLogger, dist_utils

_EXPECTED_IMAGES = 5_000
_EXPECTED_PREDICTIONS_PER_IMAGE = 300
_COCO_CATEGORY_IDS = frozenset(mscoco_category2name)


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers must not be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if dist_utils.get_world_size() != 1:
        raise RuntimeError("D-FINE reproduction requires a single process")

    cfg = YAMLConfig(str(args.config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    val = cfg.yaml_cfg["val_dataloader"]
    val["total_batch_size"] = args.batch_size
    val["num_workers"] = args.num_workers
    dataset = val["dataset"]
    dataset["img_folder"] = str(args.images)
    dataset["ann_file"] = str(args.annotations)

    if cfg.yaml_cfg.get("remap_mscoco_category") is not True:
        raise RuntimeError("COCO category remapping is not enabled")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if set(checkpoint) != {"model"}:
        raise RuntimeError(
            f"unexpected released checkpoint structure: {sorted(checkpoint)}"
        )

    state = checkpoint["model"]
    if not isinstance(state, dict) or not state:
        raise RuntimeError("released model state is empty")

    model = cfg.model
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda")
    model = model.to(device).eval()
    postprocessor = cfg.postprocessor.to(device).eval()
    data_loader = cfg.val_dataloader

    if len(data_loader.dataset) != _EXPECTED_IMAGES:
        raise RuntimeError(
            f"unexpected validation image count: {len(data_loader.dataset)}"
        )

    images, predictions = _run(
        model,
        postprocessor,
        data_loader,
        device,
        args.output,
    )

    expected = _EXPECTED_IMAGES * _EXPECTED_PREDICTIONS_PER_IMAGE
    if images != _EXPECTED_IMAGES:
        raise RuntimeError(f"unexpected processed image count: {images}")
    if predictions != expected:
        raise RuntimeError(
            f"unexpected prediction count: {predictions}; expected {expected}"
        )

    print(f"images       {images}")
    print(f"predictions  {predictions}")
    print(f"output       {args.output}")


@torch.no_grad()
def _run(
    model,
    postprocessor,
    data_loader,
    device,
    output: Path,
) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_logger = MetricLogger(delimiter=" ")
    image_count = 0
    prediction_count = 0
    temporary: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write("[")
            first_image = True

            for samples, targets in metric_logger.log_every(
                data_loader, 10, "Test:"
            ):
                samples = samples.to(device)
                targets = [
                    {
                        key: value.to(device)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in target.items()
                    }
                    for target in targets
                ]

                outputs = model(samples)
                orig_sizes = torch.stack(
                    [target["orig_size"] for target in targets], dim=0
                )
                results = postprocessor(outputs, orig_sizes)
                if len(results) != len(targets):
                    raise RuntimeError(
                        "postprocessor result count does not match batch size"
                    )

                for target, result in zip(targets, results, strict=True):
                    if not first_image:
                        file.write(",")
                    image_id = int(target["image_id"].item())
                    prediction_count += _write_result(file, image_id, result)
                    image_count += 1
                    first_image = False

            file.write("]\n")

        temporary.replace(output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    return image_count, prediction_count


def _write_result(file: TextIO, image_id: int, result, /) -> int:
    labels = result["labels"].detach().cpu()
    boxes = result["boxes"].detach().cpu().clone()
    scores = result["scores"].detach().cpu()

    if (
        labels.shape != (_EXPECTED_PREDICTIONS_PER_IMAGE,)
        or boxes.shape != (_EXPECTED_PREDICTIONS_PER_IMAGE, 4)
        or scores.shape != (_EXPECTED_PREDICTIONS_PER_IMAGE,)
    ):
        raise RuntimeError(
            f"unexpected postprocessor shapes for image {image_id}"
        )
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise RuntimeError(f"non-finite predictions for image {image_id}")
    if (scores < 0).any() or (scores > 1).any():
        raise RuntimeError(f"scores outside [0, 1] for image {image_id}")

    observed = {int(label) for label in labels.tolist()}
    unexpected = observed - _COCO_CATEGORY_IDS
    if unexpected:
        raise RuntimeError(
            f"unexpected COCO category IDs: {sorted(unexpected)}"
        )

    boxes[:, 2:] -= boxes[:, :2]
    records = [
        {
            "image_id": image_id,
            "category_id": label,
            "bbox": box,
            "score": score,
        }
        for label, box, score in zip(
            labels.tolist(), boxes.tolist(), scores.tolist(), strict=True
        )
    ]
    encoded = json.dumps(records, allow_nan=False, separators=(",", ":"))
    file.write(encoded[1:-1])
    return len(records)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    main()
