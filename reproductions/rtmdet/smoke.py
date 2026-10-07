import platform

import mmcv
import mmengine
import numpy
import setuptools
import torch
import torchvision
from mmcv.ops import nms
from mmengine.utils.dl_utils import collect_env


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    _ = collect_env()

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]],
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8], device="cuda")
    _, keep = nms(boxes, scores, 0.5)
    indices = keep.tolist()
    if indices != [0]:
        raise RuntimeError(f"unexpected MMCV NMS result: {indices!r}")

    print(f"python       {platform.python_version()}")
    print(f"numpy        {numpy.__version__}")
    print(f"torch        {torch.__version__}")
    print(f"torch cuda   {torch.version.cuda}")
    print(f"torchvision  {torchvision.__version__}")
    print(f"mmcv         {mmcv.__version__}")
    print(f"mmengine     {mmengine.__version__}")
    print(f"setuptools   {setuptools.__version__}")
    print(f"device       {torch.cuda.get_device_name()}")
    print("mmengine env ok")
    print("mmcv ops     ok")


if __name__ == "__main__":
    main()
