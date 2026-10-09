import platform

import mmcv
import mmdet
import mmengine
import numpy
import setuptools
import torch
import torchvision
from mmcv.ops import nms
from mmdet.utils.collect_env import collect_env


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device("cuda")
    environment = collect_env()
    if environment.get("CUDA available") is not True:
        raise RuntimeError("MMDetection environment does not report CUDA")
    if "MMDetection" not in environment:
        raise RuntimeError("MMDetection environment information is incomplete")

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
        device=device,
    )
    scores = torch.tensor([0.9, 0.8], device=device)
    _, keep = nms(boxes, scores, 0.5)
    torch.cuda.synchronize(device)

    indices = keep.cpu().tolist()
    if indices != [0]:
        raise RuntimeError(f"unexpected MMCV NMS result: {indices!r}")

    values = (
        ("python", platform.python_version()),
        ("numpy", numpy.__version__),
        ("torch", torch.__version__),
        ("torch cuda", str(torch.version.cuda)),
        ("torchvision", torchvision.__version__),
        ("mmcv", mmcv.__version__),
        ("mmengine", mmengine.__version__),
        ("mmdet", mmdet.__version__),
        ("setuptools", setuptools.__version__),
        ("device", torch.cuda.get_device_name(device)),
        ("mmdet env", "ok"),
        ("mmcv ops", "ok"),
    )
    for label, value in values:
        print(f"{label:<13}{value}")


if __name__ == "__main__":
    main()
