"""第 0 阶段环境冒烟测试；仅验证 CPU 计算及自动求导。"""

import torch


def test_cpu_matrix_multiplication_and_autograd():
    x = torch.arange(6, dtype=torch.float32, device="cpu").reshape(2, 3)
    x.requires_grad_()
    weight = torch.ones((3, 4), device="cpu")
    y = x @ weight
    torch.testing.assert_close(
        y, torch.tensor([[3.0] * 4, [12.0] * 4], device="cpu")
    )
    y.sum().backward()
    assert y.device.type == "cpu"
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))
