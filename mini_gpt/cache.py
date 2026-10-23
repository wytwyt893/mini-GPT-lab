"""第 9 章练习：逐层保存并追加注意力的 K/V 张量。"""

import torch


# 缓存是一对张量 (K, V)，每个张量的形状都是 [B,H,历史长度,D]。
KVCache = tuple[torch.Tensor, torch.Tensor]


def append_kv_cache(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    past_kv: KVCache | None = None,
) -> KVCache:
    """把当前 K/V [B,H,T,D] 接到历史缓存 [B,H,P,D] 的时间维。

    P 为历史长度，T 为当前输入长度，返回两份 [B,H,P+T,D]。
    首次调用 past_kv=None 时，当前 K/V 就是第一份缓存。
    此函数由 Attention 调用来追加张量，分数与因果遮罩在 Attention 中计算。
    """
    if k_new.ndim != 4 or k_new.shape != v_new.shape:
        raise ValueError("当前 K 和 V 必须是形状相同的四维张量 [B,H,T,D]")

    # 首次调用：当前 K/V 就是第一份缓存。
    if past_kv is None:
        return k_new, v_new

    # 取出历史 K/V。
    past_k, past_v = past_kv

    # 沿时间维追加，维度顺序为 [B,H,T,D]。
    k_all = torch.cat((past_k, k_new), dim=2)
    v_all = torch.cat((past_v, v_new), dim=2)

    return k_all, v_all
