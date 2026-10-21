"""验证 FFN 的数值、非线性、位置独立性、参数共享与梯度。"""

import pytest
import torch

from mini_gpt.blocks import FeedForward


def test_ffn_matches_known_nonlinear_result():
    layer = FeedForward(n_embd=1)
    with torch.no_grad():
        layer.up_proj.weight.copy_(torch.tensor([[2.], [-1.], [0.], [0.]]))
        layer.up_proj.bias.copy_(torch.tensor([1., 0., 0., 0.]))
        layer.down_proj.weight.copy_(torch.tensor([[0.5, 0., 0., 0.]]))
        layer.down_proj.bias.fill_(0.25)

    # 手算 y=0.5*GELU(2x+1)+0.25，独立数值可检查激活位置及偏置处理。
    x = torch.tensor([[[-1.], [0.], [1.]]])
    expected = torch.tensor([[[0.17067237], [0.67067237], [1.74797515]]])
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_changing_one_position_preserves_other_outputs():
    layer = FeedForward(n_embd=4)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    changed_x = x.clone()
    changed_x[:, 1] += 5.0
    output, changed_output = layer(x), layer(changed_x)
    assert output.shape == x.shape
    torch.testing.assert_close(output[:, [0, 2]], changed_output[:, [0, 2]])


def test_identical_vectors_use_shared_parameters_across_positions():
    layer = FeedForward(n_embd=4)
    x = torch.tensor([[[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 3., 4.]]])
    output = layer(x)
    torch.testing.assert_close(output[:, 0], output[:, 2])


def test_ffn_supports_backward():
    layer = FeedForward(n_embd=2)
    with torch.no_grad():
        layer.up_proj.weight.fill_(0.25)
        layer.up_proj.bias.fill_(0.1)
        layer.down_proj.weight.fill_(0.5)
        layer.down_proj.bias.fill_(0.1)
    x = torch.tensor([[[-1., 2.], [0.5, -0.25]]], requires_grad=True)
    layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("n_embd", [0, -1])
def test_invalid_hidden_dimension_is_rejected(n_embd):
    with pytest.raises(ValueError, match="隐藏维度必须为正数"):
        FeedForward(n_embd=n_embd)
