"""逐位置前馈网络与第 6 章 Pre-LN 解码器块练习。"""

import torch
from torch import nn

from mini_gpt.attention import CausalSelfAttention
from mini_gpt.cache import KVCache

class FeedForward(nn.Module):
    """每个位置使用同一组参数，完成 C -> 4C -> C 的非线性变换。
    参数：
        n_embd：隐藏维度 C
    数据流举例：
        假设输入 x 的形状为 [2, 3, 4]，即 B=2, T=3, C=4。
        升维后为 [2, 3, 16]，GELU激活后仍为 [2, 3, 16]，降维后回到 [2, 3, 4]。
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        if n_embd <= 0:
            raise ValueError("隐藏维度必须为正数")

        # 建立升维线性层 self.up_proj，C -> 4C。
        self.up_proj = nn.Linear(n_embd, 4 * n_embd)
        # 建立 GELU 激活 self.act。
        self.act = nn.GELU()
        # 建立降维线性层 self.down_proj，4C -> C。
        self.down_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入、输出均为浮点张量 [B,T,C]，各位置之间不交换信息。"""
        # 按“升维 -> 激活 -> 降维”的顺序调用子层并返回结果。
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        return x


class DecoderBlock(nn.Module):
    """组合归一化、因果注意力、FFN 和残差；输入输出均为 [B,T,C]。"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        # 建立第一次归一化 self.ln_1。
        self.ln_1 = nn.LayerNorm(n_embd)

        # 建立注意力模块 self.attn，复用已经完成的类。
        self.attn = CausalSelfAttention(n_embd, n_head)

        # 建立第二次归一化 self.ln_2，与第一次使用独立参数。
        self.ln_2 = nn.LayerNorm(n_embd)

        # 建立前馈模块 self.ffn，复用上面的 FeedForward。
        self.ffn = FeedForward(n_embd)

    def forward(
    self,
    x: torch.Tensor,
    past_kv: KVCache | None = None,
    use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """输入输出均为 [B,T,C]；可同时返回本层的累积 K/V。"""
        attn_input = self.ln_1(x)

        # 把本层历史缓存交给 Attention。
        attn_result = self.attn(
            attn_input,
            past_kv=past_kv,
            use_cache=use_cache,
        )

        # Attention 在启用缓存时返回两个结果。
        if use_cache:
            attn_update, present_kv = attn_result
        else:
            attn_update = attn_result

        # 第一次残差只更新本次输入的位置。
        h = x + attn_update

        ffn_input = self.ln_2(h)
        ffn_update = self.ffn(ffn_input)

        # 第二次残差同样保持 [B,T,C]。
        y = h + ffn_update

        if use_cache:
            return y, present_kv
        return y
