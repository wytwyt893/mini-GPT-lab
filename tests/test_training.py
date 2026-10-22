"""验证单步更新、历史梯度清理、循环记录及优化器状态延续。"""

import copy

import pytest
import torch

from mini_gpt.losses import make_next_token_batch, next_token_loss
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer
from scripts.train_tiny import train_loop, train_step


def training_batch():
    tokenizer = CharTokenizer("abc ")
    token_ids = torch.tensor([tokenizer.encode("abc abc a")], dtype=torch.long, device="cpu")
    return make_next_token_batch(token_ids)


def test_step_returns_previous_loss_updates_parameters_and_lowers_batch_loss():
    torch.manual_seed(42)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu").eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    inputs, targets = training_batch()
    before_weight = model.lm_head.weight.detach().clone()
    with torch.no_grad():
        before_loss = next_token_loss(model(inputs), targets).item()

    returned_loss = train_step(model, optimizer, inputs, targets)
    assert model.training
    assert isinstance(returned_loss, float)
    assert returned_loss == pytest.approx(before_loss)
    assert not torch.equal(model.lm_head.weight, before_weight)
    model.eval()
    with torch.no_grad():
        after_loss = next_token_loss(model(inputs), targets).item()
    # 此断言只针对这批固定数据与明确的初始化、学习率，不假设所有更新都使 loss 单调下降。
    assert after_loss < before_loss


def test_step_clears_stale_gradients_before_backward():
    torch.manual_seed(43)
    clean_model = MiniGPT(4, 8, 8, 2, 2).to("cpu")
    stale_model = copy.deepcopy(clean_model)
    for parameter in stale_model.parameters():
        parameter.grad = torch.full_like(parameter, 100.0)
    clean_optimizer = torch.optim.SGD(clean_model.parameters(), lr=0.05)
    stale_optimizer = torch.optim.SGD(stale_model.parameters(), lr=0.05)
    inputs, targets = training_batch()

    clean_loss = train_step(clean_model, clean_optimizer, inputs, targets)
    stale_loss = train_step(stale_model, stale_optimizer, inputs, targets)
    assert stale_loss == pytest.approx(clean_loss)
    for clean_parameter, stale_parameter in zip(clean_model.parameters(), stale_model.parameters()):
        torch.testing.assert_close(clean_parameter, stale_parameter)


def test_loop_preserves_loss_order_reuses_objects_and_logs_checkpoints(monkeypatch, capsys):
    model = MiniGPT(4, 8, 4, 2, 1).to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs, targets = training_batch()
    # 故意使用非单调序列，验证记录的是逐步原始结果，而非排序或平滑后的值。
    expected_losses = [3.0, 2.0, 4.0] + [1.0] * 18
    remaining_losses = iter(expected_losses)
    calls = []

    def fake_step(actual_model, actual_optimizer, actual_inputs, actual_targets):
        assert actual_model is model
        assert actual_optimizer is optimizer
        assert actual_inputs is inputs
        assert actual_targets is targets
        calls.append(1)
        return next(remaining_losses)

    monkeypatch.setattr("scripts.train_tiny.train_step", fake_step)
    history = train_loop(model, optimizer, inputs, targets, num_steps=21)
    assert history == expected_losses
    assert len(calls) == 21
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    for line, marker in zip(lines, ("1/21", "20/21", "21/21")):
        assert marker in line


def test_loop_continues_real_optimizer_state_across_updates():
    torch.manual_seed(42)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    inputs, targets = training_batch()
    before_weight = model.lm_head.weight.detach().clone()
    history = train_loop(model, optimizer, inputs, targets, num_steps=3)
    assert len(history) == 3
    assert all(isinstance(value, float) for value in history)
    assert torch.isfinite(torch.tensor(history)).all()
    assert optimizer.state[model.lm_head.weight]["step"].item() == 3
    assert not torch.equal(model.lm_head.weight, before_weight)


@pytest.mark.parametrize("num_steps", [0, -1])
def test_loop_rejects_nonpositive_step_count(num_steps):
    model = MiniGPT(4, 8, 4, 2, 1).to("cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs, targets = training_batch()
    with pytest.raises(ValueError, match="训练步数必须为正数"):
        train_loop(model, optimizer, inputs, targets, num_steps=num_steps)
