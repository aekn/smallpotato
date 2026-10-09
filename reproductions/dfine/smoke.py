import argparse
import platform
from pathlib import Path

import numpy
import torch
import torchvision
from src.core import YAMLConfig
from src.data.dataset import mscoco_category2name

_IMAGE_SIZE = 640
_NUM_DETECTIONS = 300


def main() -> None:
    config, checkpoint_path = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    cfg = YAMLConfig(str(config))
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    checkpoint = torch.load(
        checkpoint_path,
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
    images = torch.zeros(
        (1, 3, _IMAGE_SIZE, _IMAGE_SIZE),
        device=device,
    )
    orig_sizes = torch.tensor(
        [[_IMAGE_SIZE, _IMAGE_SIZE]],
        device=device,
    )

    with torch.no_grad():
        results = postprocessor(model(images), orig_sizes)
    torch.cuda.synchronize(device)

    if len(results) != 1:
        raise RuntimeError(f"unexpected result count: {len(results)}")
    result = results[0]
    labels = result["labels"]
    boxes = result["boxes"]
    scores = result["scores"]

    if (
        labels.shape != (_NUM_DETECTIONS,)
        or boxes.shape != (_NUM_DETECTIONS, 4)
        or scores.shape != (_NUM_DETECTIONS,)
    ):
        raise RuntimeError(
            "unexpected postprocessor shapes: "
            f"labels={tuple(labels.shape)}, boxes={tuple(boxes.shape)}, "
            f"scores={tuple(scores.shape)}"
        )
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
        raise RuntimeError("non-finite postprocessor output")
    if (scores < 0).any() or (scores > 1).any():
        raise RuntimeError("postprocessor scores are outside [0, 1]")

    category_ids = set(mscoco_category2name)
    observed = {int(value) for value in labels.tolist()}
    if not observed <= category_ids:
        raise RuntimeError(
            f"unexpected COCO category IDs: {sorted(observed - category_ids)}"
        )

    values = (
        ("python", platform.python_version()),
        ("numpy", numpy.__version__),
        ("torch", torch.__version__),
        ("torch cuda", str(torch.version.cuda)),
        ("torchvision", torchvision.__version__),
        ("device", torch.cuda.get_device_name(device)),
        ("state tensors", str(len(state))),
        ("checkpoint", "ok"),
        ("dfine model", "ok"),
    )
    for label, value in values:
        print(f"{label:<14}{value}")


def _parse_args() -> tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    return args.config, args.checkpoint


if __name__ == "__main__":
    main()
