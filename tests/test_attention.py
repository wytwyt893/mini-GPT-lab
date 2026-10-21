"""验证单头及多头的数值、批量独立性、因果性和梯度。"""

import pytest

import torch

from mini_gpt.attention import CausalSelfAttention


def identity_attention(n_embd: int = 2, n_head: int = 1) -> CausalSelfAttention:
    """用恒等投影构造能独立手算结果的小型层。"""
    layer = CausalSelfAttention(n_embd=n_embd, n_head=n_head)
    with torch.no_grad():
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            projection.weight.copy_(torch.eye(n_embd))
            projection.bias.zero_()
    return layer


def test_single_head_matches_hand_calculated_result():
    layer = identity_attention()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    # 手算分数除以 sqrt(2)，应用因果遮罩并加权求和后的结果。
    expected = torch.tensor([[[1.0, 0.0], [0.330238, 0.669762], [0.751745, 0.751745]]])
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_future_input_does_not_change_prefix_outputs():
    layer = identity_attention()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]]])
    changed_x = x.clone()
    changed_x[:, 2:] += 10.0
    torch.testing.assert_close(layer(x)[:, :2], layer(changed_x)[:, :2])


def test_samples_in_a_batch_do_not_mix():
    layer = identity_attention()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [3.0, 4.0]]])
    batched = layer(x)
    assert batched.shape == (2, 2, 2)
    for batch_index in range(2):
        torch.testing.assert_close(
            batched[batch_index:batch_index + 1], layer(x[batch_index:batch_index + 1])
        )


def test_single_head_supports_backward():
    layer = identity_attention()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0


def test_two_heads_match_hand_calculation():
    layer = identity_attention(n_embd=4, n_head=2)
    x = torch.tensor([[[1., 0., 0., 1.], [1., 0., 1., 0.]]])
    # 第二个位置：头 0 的权重为 [0.5,0.5]，头 1 为 [0.330238,0.669762]。
    # 直接使用手算的拼接结果，能区分错误的单头实现及错误的缩放维度。
    expected = torch.tensor([[[1., 0., 0., 1.], [1., 0., 0.669762, 0.330238]]])
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)


def test_two_heads_preserve_causality():
    layer = identity_attention(n_embd=4, n_head=2)
    x = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4) / 10
    changed_x = x.clone()
    changed_x[:, 2:] += 10.0
    torch.testing.assert_close(layer(x)[:, :2], layer(changed_x)[:, :2])


@pytest.mark.parametrize("n_head", [1, 2, 4])
def test_heads_preserve_batch_independence(n_head):
    layer = identity_attention(n_embd=4, n_head=n_head)
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    batched = layer(x)
    assert batched.shape == x.shape
    for batch_index in range(2):
        torch.testing.assert_close(
            batched[batch_index:batch_index + 1], layer(x[batch_index:batch_index + 1])
        )


def test_two_heads_support_backward():
    layer = identity_attention(n_embd=4, n_head=2)
    x = torch.tensor([[[1., 0., 0., 1.], [0., 1., 2., 0.], [1., 1., 1., 1.]]],
                     requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("n_embd,n_head", [(4, 3), (0, 1), (4, 0)])
def test_invalid_head_configuration_is_rejected(n_embd, n_head):
    with pytest.raises(ValueError):
        CausalSelfAttention(n_embd=n_embd, n_head=n_head)
