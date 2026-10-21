"""验证嵌入层的位置相加、批量广播、上下文边界及梯度。"""

import pytest
import torch

from mini_gpt.embeddings import TokenPositionEmbedding


def test_known_vectors_and_positions_across_batch():
    layer = TokenPositionEmbedding(vocab_size=3, block_size=4, n_embd=2)
    with torch.no_grad():
        layer.token_embedding.weight.copy_(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        )
        layer.position_embedding.weight.copy_(
            torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
        )

    output = layer(torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    # 用手算的具体数值验证两条序列均从位置 0 开始，不发生跨样本混合。
    expected = torch.tensor([
        [[11.0, 22.0], [33.0, 44.0]],
        [[13.0, 24.0], [31.0, 42.0]],
    ])
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("length", [1, 4])
def test_output_shape_and_context_boundary(length):
    layer = TokenPositionEmbedding(vocab_size=3, block_size=4, n_embd=2)
    output = layer(torch.zeros((2, length), dtype=torch.long))
    assert output.shape == (2, length, 2)
    assert output.dtype == torch.float32


def test_overlong_sequence_is_rejected():
    layer = TokenPositionEmbedding(vocab_size=3, block_size=4, n_embd=2)
    with pytest.raises(ValueError, match="输入序列超过最大上下文长度"):
        layer(torch.zeros((1, 5), dtype=torch.long))


def test_gradients_accumulate_into_used_rows():
    layer = TokenPositionEmbedding(vocab_size=3, block_size=4, n_embd=2)
    output = layer(torch.tensor([[0, 1, 0]], dtype=torch.long))
    output.sum().backward()

    # 字符 0 使用两次，字符 1 使用一次，字符 2 未使用。
    assert layer.token_embedding.weight.grad is not None
    torch.testing.assert_close(
        layer.token_embedding.weight.grad,
        torch.tensor([[2.0, 2.0], [1.0, 1.0], [0.0, 0.0]]),
    )
    # 位置 0、1、2 各使用一次，位置 3 未使用。
    assert layer.position_embedding.weight.grad is not None
    torch.testing.assert_close(
        layer.position_embedding.weight.grad,
        torch.tensor([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
    )
