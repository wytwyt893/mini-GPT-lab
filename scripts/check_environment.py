"""使用微型 CPU 自动求导计算验证学习环境。"""

import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch


def main() -> None:
    device = torch.device("cpu")
    x = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    x.requires_grad_()
    weight = torch.ones((3, 4), device=device)
    y = x @ weight
    y.sum().backward()
    assert y.shape == (2, 4)
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))

    report = {
        "project_root": str(Path(__file__).resolve().parents[1]),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "environment_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "torch": torch.__version__,
        "torch_location": torch.__file__,
        "pytest": version("pytest"),
        "device": str(y.device),
        "input_shape": list(x.shape),
        "weight_shape": list(weight.shape),
        "output_shape": list(y.shape),
        "cpu_forward_backward": "passed",
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
