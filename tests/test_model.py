"""验证最小 GPT 的前向数值、堆叠因果性、长度边界与梯度。"""

import math

import pytest
import torch

from mini_gpt.model import MiniGPT


@pytest.mark.parametrize("n_layer,batch_size,length", [(1, 1, 1), (2, 2, 3), (2, 1, 8)])
def test_model_outputs_one_score_per_vocabulary_entry(n_layer, batch_size, length):
    model = MiniGPT(4, 8, 8, 2, n_layer).to("cpu")
    token_ids = torch.arange(batch_size * length, device="cpu").reshape(batch_size, length) % 4
    logits = model(token_ids)
    assert logits.shape == (batch_size, length, 4)
    assert logits.dtype == torch.float32
    assert logits.device.type == "cpu"
    assert torch.isfinite(logits).all()


def test_final_normalization_and_lm_head_match_hand_calculation():
    model = MiniGPT(3, 2, 2, 1, 2).to("cpu")
    with torch.no_grad():
        model.embedding.token_embedding.weight.copy_(torch.tensor([[1., 3.], [3., 1.], [2., 2.]]))
        model.embedding.position_embedding.weight.zero_()
        # 关闭各 Block 的更新量，使残差主干保留嵌入向量，便于独立检查最终 LN 与输出头。
        for block in model.blocks:
            for projection in (block.attn.out_proj, block.ffn.down_proj):
                projection.weight.zero_()
                projection.bias.zero_()
        model.ln_f.weight.copy_(torch.tensor([2., 0.5]))
        model.ln_f.bias.copy_(torch.tensor([0.25, -0.75]))
        model.lm_head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., -1.]]))

    a = 1.0 / math.sqrt(1.0 + 1e-5)
    # 两个输入归一化前的均值均为 2、方差均为 1；独立手算三个词表分数。
    expected = torch.tensor([[[-2*a + 0.25, 0.5*a - 0.75, -2.5*a + 1.0],
                              [2*a + 0.25, -0.5*a - 0.75, 2.5*a + 1.0]]])
    torch.testing.assert_close(model(torch.tensor([[0, 1]], device="cpu")), expected)


def test_stacked_model_preserves_causality():
    torch.manual_seed(31)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu").eval()
    token_ids = torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1]], device="cpu")
    changed_ids = token_ids.clone()
    changed_ids[:, 2:] = (changed_ids[:, 2:] + 1) % 4
    logits, changed_logits = model(token_ids), model(changed_ids)
    torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
    torch.testing.assert_close(logits[:, :2], model(token_ids[:, :2]))
    assert not torch.allclose(logits[:, 2:], changed_logits[:, 2:])


def test_model_preserves_batch_independence():
    torch.manual_seed(37)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu").eval()
    token_ids = torch.tensor([[0, 1, 2], [3, 2, 0]], device="cpu")
    logits = model(token_ids)
    for index in range(2):
        torch.testing.assert_close(logits[index:index + 1], model(token_ids[index:index + 1]))


def test_gradients_reach_embeddings_every_block_and_lm_head():
    torch.manual_seed(41)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu")
    # 此标量只用于检查反向路径；本节尚未实现语言模型训练损失。
    model(torch.tensor([[0, 1, 2], [3, 2, 0]], device="cpu")).square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    weights = [model.embedding.token_embedding.weight,
               model.embedding.position_embedding.weight,
               model.ln_f.weight, model.lm_head.weight]
    weights.extend(block.attn.out_proj.weight for block in model.blocks)
    for weight in weights:
        assert weight.grad.abs().sum() > 0


def test_model_rejects_input_longer_than_context():
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu")
    with pytest.raises(ValueError, match="输入序列超过最大上下文长度"):
        model(torch.zeros((1, 9), dtype=torch.long, device="cpu"))


@pytest.mark.parametrize("n_layer", [0, -1])
def test_model_rejects_nonpositive_layer_count(n_layer):
    with pytest.raises(ValueError, match="解码器层数必须为正数"):
        MiniGPT(4, 8, 8, 2, n_layer)
