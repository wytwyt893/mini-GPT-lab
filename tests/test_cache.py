"""验证 K/V 追加以及缓存 Attention 的数值一致性和因果性。"""

import pytest
import torch

from mini_gpt.attention import CausalSelfAttention
from mini_gpt.cache import append_kv_cache


def test_first_cache_preserves_current_keys_and_values():
    keys = torch.arange(24, dtype=torch.float32, device="cpu").reshape(1, 2, 3, 4)
    values = -keys - 1.0
    cached_keys, cached_values = append_kv_cache(keys, values)
    assert cached_keys.shape == cached_values.shape == (1, 2, 3, 4)
    torch.testing.assert_close(cached_keys, keys)
    torch.testing.assert_close(cached_values, values)


@pytest.mark.parametrize("new_length", [1, 2])
def test_append_preserves_past_and_adds_new_positions_for_every_batch_and_head(new_length):
    past_keys = torch.arange(24, dtype=torch.float32, device="cpu").reshape(2, 2, 2, 3)
    past_values = -past_keys - 100.0
    keys = torch.arange(12 * new_length, dtype=torch.float32).reshape(2, 2, new_length, 3) + 1000.0
    values = -keys - 100.0
    original_keys, original_values = past_keys.clone(), past_values.clone()

    cached_keys, cached_values = append_kv_cache(keys, values, (past_keys, past_values))
    assert cached_keys.shape == cached_values.shape == (2, 2, 2 + new_length, 3)
    torch.testing.assert_close(cached_keys[:, :, :2], past_keys)
    torch.testing.assert_close(cached_values[:, :, :2], past_values)
    torch.testing.assert_close(cached_keys[:, :, 2:], keys)
    torch.testing.assert_close(cached_values[:, :, 2:], values)
    torch.testing.assert_close(past_keys, original_keys)
    torch.testing.assert_close(past_values, original_values)


@pytest.mark.parametrize("key_shape,value_shape", [((2, 3, 4), (2, 3, 4)),
                                                 ((1, 2, 1, 4), (1, 2, 2, 4))])
def test_invalid_current_key_value_shapes_are_rejected(key_shape, value_shape):
    with pytest.raises(ValueError, match="形状相同的四维张量"):
        append_kv_cache(torch.zeros(key_shape), torch.zeros(value_shape))


def test_attention_prefill_returns_projected_keys_and_values_in_head_order():
    layer = CausalSelfAttention(n_embd=4, n_head=2).to("cpu").eval()
    with torch.no_grad():
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
        x = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.]]], device="cpu")
        output, (keys, values) = layer(x, use_cache=True)
        # 两个头各取每个位置的两维，时间维仍按输入顺序排列。
        expected = torch.tensor([[[[1., 2.], [5., 6.]],
                                  [[3., 4.], [7., 8.]]]], device="cpu")
        torch.testing.assert_close(keys, expected)
        torch.testing.assert_close(values, expected)
        torch.testing.assert_close(output, layer(x))


@pytest.mark.parametrize("n_head,chunk_lengths", [
    (1, (1, 1, 1, 1, 1, 1)),
    (2, (1, 1, 1, 1, 1, 1)),
    (2, (2, 3, 1)),
    (4, (3, 3)),
])
def test_cached_attention_matches_full_sequence_for_tokens_and_chunks(n_head, chunk_lengths):
    torch.manual_seed(42)
    layer = CausalSelfAttention(n_embd=8, n_head=n_head).to("cpu").eval()
    x = torch.randn(2, 6, 8, device="cpu")
    with torch.no_grad():
        expected = layer(x)
        past_kv = None
        outputs = []
        start = 0
        for length in chunk_lengths:
            previous = None if past_kv is None else tuple(t.clone() for t in past_kv)
            output, updated = layer(x[:, start:start + length], past_kv=past_kv, use_cache=True)
            outputs.append(output)
            start += length
            assert updated[0].shape == updated[1].shape == (2, n_head, start, 8 // n_head)
            if previous is not None:
                for old, original, new in zip(past_kv, previous, updated):
                    torch.testing.assert_close(old, original)
                    torch.testing.assert_close(new[:, :, :start - length], original)
            past_kv = updated
        torch.testing.assert_close(torch.cat(outputs, dim=1), expected, rtol=1e-5, atol=1e-6)


def test_cached_chunk_cannot_read_later_positions_in_the_same_chunk():
    torch.manual_seed(42)
    layer = CausalSelfAttention(n_embd=8, n_head=2).to("cpu").eval()
    x = torch.randn(1, 5, 8, device="cpu")
    with torch.no_grad():
        _, past_kv = layer(x[:, :2], use_cache=True)
        current = x[:, 2:]
        changed = current.clone()
        changed[:, -1] += 10.0
        output, _ = layer(current, past_kv=past_kv, use_cache=True)
        changed_output, _ = layer(changed, past_kv=past_kv, use_cache=True)
        torch.testing.assert_close(output[:, :-1], changed_output[:, :-1])
        assert not torch.allclose(output[:, -1], changed_output[:, -1])


def test_attention_projects_only_new_positions_when_reusing_cache():
    layer = CausalSelfAttention(n_embd=8, n_head=2).to("cpu").eval()
    x = torch.randn(1, 6, 8, device="cpu")
    observed_lengths = {name: [] for name in ("q_proj", "k_proj", "v_proj")}

    def record_input_length(name):
        def hook(module, args, output):
            observed_lengths[name].append(args[0].size(1))
        return hook

    handles = [getattr(layer, name).register_forward_hook(record_input_length(name))
               for name in observed_lengths]
    try:
        with torch.no_grad():
            _, cache = layer(x[:, :3], use_cache=True)
            _, cache = layer(x[:, 3:4], past_kv=cache, use_cache=True)
            _, cache = layer(x[:, 4:], past_kv=cache, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    assert observed_lengths == {name: [3, 1, 2] for name in observed_lengths}
    assert cache[0].size(2) == cache[1].size(2) == 6


def test_existing_cache_can_be_read_without_returning_a_new_cache():
    torch.manual_seed(42)
    layer = CausalSelfAttention(n_embd=8, n_head=2).to("cpu").eval()
    x = torch.randn(1, 4, 8, device="cpu")
    with torch.no_grad():
        _, cache = layer(x[:, :3], use_cache=True)
        output = layer(x[:, 3:], past_kv=cache)
        assert isinstance(output, torch.Tensor)
        torch.testing.assert_close(output, layer(x)[:, 3:], rtol=1e-5, atol=1e-6)
