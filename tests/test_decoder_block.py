"""验证 Pre-LN 解码器块的数值、残差、因果性、批量独立性与梯度。"""

import math

import pytest
import torch

from mini_gpt.blocks import DecoderBlock


@pytest.mark.parametrize("shape", [(1, 1, 4), (2, 3, 8)])
def test_block_preserves_shape_on_cpu(shape):
    block = DecoderBlock(n_embd=shape[-1], n_head=2).to("cpu")
    x = torch.randn(shape, device="cpu")
    output = block(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()


def test_block_matches_independent_scalar_calculation():
    block = DecoderBlock(n_embd=2, n_head=1).to("cpu")
    with torch.no_grad():
        for projection in (block.attn.q_proj, block.attn.k_proj,
                           block.attn.v_proj, block.attn.out_proj):
            projection.weight.copy_(torch.eye(2))
            projection.bias.zero_()
        # 让 FFN 只处理归一化后的第一个坐标，输出 [GELU(r), -GELU(r)] 加偏置。
        block.ffn.up_proj.weight.zero_()
        block.ffn.up_proj.weight[0, 0] = 1.0
        block.ffn.up_proj.bias.zero_()
        block.ffn.down_proj.weight.zero_()
        block.ffn.down_proj.weight[0, 0] = 1.0
        block.ffn.down_proj.weight[1, 0] = -1.0
        block.ffn.down_proj.bias.copy_(torch.tensor([0.25, -0.75]))

    x = torch.tensor([[[1., 3.], [3., 1.]]], device="cpu")
    # 使用独立标量公式手算，不调用被测的 LayerNorm、Attention 或 FFN 求期望值。
    a = 1.0 / math.sqrt(1.0 + 1e-5)
    previous_weight = 1.0 / (1.0 + math.exp(2.0 * math.sqrt(2.0) * a * a))
    d = a * (1.0 - 2.0 * previous_weight)
    expected_rows = []
    for first, second in ((1.0 - a, 3.0 + a), (3.0 + d, 1.0 - d)):
        # 两个位置第一次残差后的均值都是 2，第二次 LN 的方差为 (first-2)^2。
        centered = first - 2.0
        r = centered / math.sqrt(centered * centered + 1e-5)
        gelu = 0.5 * r * (1.0 + math.erf(r / math.sqrt(2.0)))
        expected_rows.append([first + gelu + 0.25, second - gelu - 0.75])
    expected = torch.tensor([expected_rows], dtype=x.dtype, device="cpu")
    torch.testing.assert_close(block(x), expected, rtol=1e-5, atol=1e-6)


def test_zero_updates_preserve_original_input():
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu")
    with torch.no_grad():
        # 两个子层的更新量都为 0，残差主干应原样保留输入。
        for projection in (block.attn.out_proj, block.ffn.down_proj):
            projection.weight.zero_()
            projection.bias.zero_()
    x = torch.tensor([[[1., 2., 3., 4.], [2., 0., 1., -1.]]], device="cpu")
    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


def test_future_input_does_not_change_prefix_outputs():
    torch.manual_seed(11)
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu")
    x = torch.randn(2, 4, 4, device="cpu")
    changed_x = x.clone()
    # 各特征采用不同改变量，避免整体平移被 LayerNorm 抵消。
    changed_x[:, 2:] += torch.tensor([3., -2., 1., 4.])
    output, changed_output = block(x), block(changed_x)
    torch.testing.assert_close(output[:, :2], changed_output[:, :2])
    assert not torch.allclose(output[:, 2:], changed_output[:, 2:])


def test_samples_in_a_batch_do_not_mix():
    torch.manual_seed(17)
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu")
    x = torch.randn(2, 3, 4, device="cpu")
    batched = block(x)
    for batch_index in range(2):
        torch.testing.assert_close(
            batched[batch_index:batch_index + 1], block(x[batch_index:batch_index + 1])
        )


def test_block_supports_backward():
    torch.manual_seed(23)
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu")
    x = torch.randn(2, 3, 4, device="cpu", requires_grad=True)
    block(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
