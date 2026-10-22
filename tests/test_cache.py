"""验证 K/V 缓存追加的时间顺序、批量/头维度与输入边界。"""

import pytest
import torch

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
