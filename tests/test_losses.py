"""验证移位标签、交叉熵独立数值、非连续张量与梯度。"""

import math

import pytest
import torch

from mini_gpt.losses import make_next_token_batch, next_token_loss


def test_shifted_inputs_and_targets_preserve_batch_order():
    token_ids = torch.tensor([[2, 0, 1, 3, 0], [0, 2, 3, 1, 2]], device="cpu")
    inputs, targets = make_next_token_batch(token_ids)
    torch.testing.assert_close(inputs, torch.tensor([[2, 0, 1, 3], [0, 2, 3, 1]]))
    torch.testing.assert_close(targets, torch.tensor([[0, 1, 3, 0], [2, 3, 1, 2]]))
    assert inputs.dtype == targets.dtype == torch.long


def test_two_characters_produce_one_training_position():
    inputs, targets = make_next_token_batch(torch.tensor([[2, 0]], device="cpu"))
    torch.testing.assert_close(inputs, torch.tensor([[2]]))
    torch.testing.assert_close(targets, torch.tensor([[0]]))


@pytest.mark.parametrize("shape", [(3,), (1, 2, 3)])
def test_shift_rejects_wrong_rank(shape):
    with pytest.raises(ValueError, match="完整编号张量必须为二维"):
        make_next_token_batch(torch.zeros(shape, dtype=torch.long, device="cpu"))


@pytest.mark.parametrize("length", [0, 1])
def test_shift_rejects_too_short_text(length):
    with pytest.raises(ValueError, match="至少需要两个字符"):
        make_next_token_batch(torch.zeros((1, length), dtype=torch.long, device="cpu"))


def test_loss_matches_independent_probability_calculation_for_noncontiguous_views():
    log_three = math.log(3.0)
    # 转置使 logits 非连续；多批次移位切片得到的 targets 也非连续。
    logits = torch.tensor([[[log_three, 0.], [0., 0.]],
                           [[0., log_three], [log_three, 0.]]], dtype=torch.float64).transpose(0, 1)
    _, targets = make_next_token_batch(torch.tensor([[1, 0, 0], [0, 1, 1]]))
    assert not logits.is_contiguous()
    assert not targets.is_contiguous()
    # 四个位置给正确编号的概率依次为 3/4、1/4、1/2、1/4。
    expected = -sum(math.log(p) for p in (0.75, 0.25, 0.5, 0.25)) / 4
    actual = next_token_loss(logits, targets)
    assert actual.ndim == 0
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


def test_extreme_logits_produce_finite_loss():
    logits = torch.tensor([[[1000., -1000.]]], device="cpu")
    actual = next_token_loss(logits, torch.tensor([[1]], device="cpu"))
    torch.testing.assert_close(actual, torch.tensor(2000.))


def test_loss_gradient_matches_uniform_prediction():
    logits = torch.zeros((1, 1, 3), device="cpu", requires_grad=True)
    loss = next_token_loss(logits, torch.tensor([[1]], device="cpu"))
    torch.testing.assert_close(loss, torch.tensor(math.log(3.0)))
    loss.backward()
    expected_gradient = torch.tensor([[[1/3, -2/3, 1/3]]])
    torch.testing.assert_close(logits.grad, expected_gradient)
