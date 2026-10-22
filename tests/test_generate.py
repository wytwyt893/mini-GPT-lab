"""验证无缓存生成的选择、回填、上下文裁剪、批量与边界。"""

import pytest
import torch

from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


@pytest.fixture
def scripted_model(monkeypatch):
    """以可独立推导的循环编号规则替代网络，专门验证生成控制流程。"""
    model = MiniGPT(4, 3, 4, 2, 1).to("cpu").eval()
    contexts = []

    def forward(token_ids):
        contexts.append(token_ids.clone())
        batch_size, length = token_ids.shape
        logits = torch.zeros(batch_size, length, 4, device="cpu")
        # 前面位置偏向 0，最后位置偏向“末尾编号加一”，可检查是否正确取了最后位置。
        logits[:, :, 0] = 5.0
        logits[:, -1, :] = 0.0
        next_ids = (token_ids[:, -1] + 1) % 4
        logits[:, -1, :].scatter_(1, next_ids.unsqueeze(-1), 10.0)
        return logits

    monkeypatch.setattr(model, "forward", forward)
    return model, contexts


def test_generated_tokens_are_fed_back_and_context_is_cropped(scripted_model):
    model, contexts = scripted_model
    prompt = torch.tensor([[0, 1], [2, 3]], device="cpu")
    output = model.generate(prompt, max_new_tokens=4)
    torch.testing.assert_close(output, torch.tensor([[0, 1, 2, 3, 0, 1], [2, 3, 0, 1, 2, 3]]))
    expected_contexts = [
        [[0, 1], [2, 3]],
        [[0, 1, 2], [2, 3, 0]],
        [[1, 2, 3], [3, 0, 1]],
        [[2, 3, 0], [0, 1, 2]],
    ]
    assert len(contexts) == 4
    for actual, expected in zip(contexts, expected_contexts):
        torch.testing.assert_close(actual, torch.tensor(expected))
    torch.testing.assert_close(prompt, torch.tensor([[0, 1], [2, 3]]))


def test_long_prompt_keeps_full_prefix_but_only_forwards_recent_window(scripted_model):
    model, contexts = scripted_model
    prompt = torch.tensor([[0, 1, 2, 3, 0]], device="cpu")
    output = model.generate(prompt, max_new_tokens=1)
    torch.testing.assert_close(contexts[0], torch.tensor([[2, 3, 0]]))
    torch.testing.assert_close(output, torch.tensor([[0, 1, 2, 3, 0, 1]]))


def test_zero_new_tokens_preserves_prompt_without_forward(scripted_model):
    model, contexts = scripted_model
    prompt = torch.tensor([[0, 1]], device="cpu")
    torch.testing.assert_close(model.generate(prompt, 0), prompt)
    assert contexts == []


@pytest.mark.parametrize("shape", [(2,), (1, 0), (0, 2)])
def test_invalid_prompt_shape_is_rejected(scripted_model, shape):
    model, contexts = scripted_model
    with pytest.raises(ValueError, match="生成输入必须为非空二维"):
        model.generate(torch.zeros(shape, dtype=torch.long), 1)
    assert contexts == []


def test_negative_generation_length_is_rejected(scripted_model):
    model, contexts = scripted_model
    with pytest.raises(ValueError, match="新增 token 数量不能为负数"):
        model.generate(torch.tensor([[0]]), -1)
    assert contexts == []


def test_real_model_reproduces_seeded_generation_without_gradient_tracking():
    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu").eval()
    prompt = torch.tensor([tokenizer.encode("cab a")], dtype=torch.long, device="cpu")
    gradient_modes = []
    handle = model.register_forward_hook(lambda module, inputs, output: gradient_modes.append(torch.is_grad_enabled()))
    try:
        output = model.generate(prompt, max_new_tokens=6)
    finally:
        handle.remove()
    assert output.shape == (1, 11)
    assert output.dtype == torch.long
    assert output.device.type == "cpu"
    assert tokenizer.decode(output[0].tolist()) == "cab abcabcb"
    assert gradient_modes == [False] * 6
