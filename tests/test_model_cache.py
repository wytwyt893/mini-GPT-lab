"""验证位置偏移、Block/GPT 缓存及缓存生成。"""

import pytest
import torch

from mini_gpt.blocks import DecoderBlock
from mini_gpt.embeddings import TokenPositionEmbedding
from mini_gpt.model import MiniGPT


def make_model():
    torch.manual_seed(42)
    return MiniGPT(
        vocab_size=4,
        block_size=8,
        n_embd=8,
        n_head=2,
        n_layer=2,
    ).to("cpu").eval()


@torch.no_grad()
def test_position_offset_uses_the_correct_embedding_row():
    embedding = TokenPositionEmbedding(4, 8, 2).to("cpu")

    embedding.token_embedding.weight[0].copy_(torch.tensor([1., 2.]))
    embedding.position_embedding.weight[3].copy_(torch.tensor([70., 80.]))

    ids = torch.tensor([[0]], dtype=torch.long, device="cpu")
    actual = embedding(ids, position_offset=3)

    # 字符编号 0，加上位置编号 3 对应的向量。
    expected = torch.tensor([[[71., 82.]]])
    torch.testing.assert_close(actual, expected)

    with pytest.raises(ValueError):
        embedding(ids, position_offset=-1)

    with pytest.raises(ValueError):
        embedding(ids, position_offset=8)


@torch.no_grad()
def test_block_cached_chunks_match_full_output():
    torch.manual_seed(42)
    block = DecoderBlock(8, 2).to("cpu").eval()
    x = torch.randn(2, 5, 8, device="cpu")

    expected = block(x)
    first, cache = block(x[:, :2], use_cache=True)
    rest, cache = block(x[:, 2:], past_kv=cache, use_cache=True)

    actual = torch.cat((first, rest), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_gpt_cached_chunks_match_full_logits():
    model = make_model()
    ids = torch.tensor([[2, 0, 1, 3, 0]], device="cpu")

    expected = model(ids)
    first, cache = model(ids[:, :3], use_cache=True)
    rest, cache = model(
        ids[:, 3:],
        past_key_values=cache,
        use_cache=True,
    )

    actual = torch.cat((first, rest), dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    assert len(cache) == 2
    for k, v in cache:
        assert k.shape == v.shape == (1, 2, 5, 4)

    # 两层拥有各自独立的缓存张量。
    assert cache[0][0].data_ptr() != cache[1][0].data_ptr()


@pytest.mark.parametrize("prompt_length,steps", [
    (3, 4),   # 窗口未满
    (5, 7),   # 生成过程中窗口滑动
    (8, 3),   # 初始提示词已填满窗口
    (11, 3),  # 初始提示词超过窗口
    (1, 0),   # 不生成新字符
])
@torch.no_grad()
def test_cached_generation_matches_original(prompt_length, steps):
    model = make_model()
    prompt = torch.randint(0, 4, (2, prompt_length), device="cpu")

    expected = model.generate(prompt, steps)
    actual = model.generate_cached(prompt, steps)

    assert torch.equal(actual, expected)