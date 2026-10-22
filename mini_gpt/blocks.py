"""逐位置前馈网络与第 6 章 Pre-LN 解码器块练习。"""

import torch
from torch import nn

from mini_gpt.attention import CausalSelfAttention


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用 Pre-LN：先归一化，再计算子层，最后加回原来的表示。"""
        # 归一化后送入 Attention，计算上下文更新量。
        attn_input = self.ln_1(x)
        attn_update = self.attn(attn_input)

        # 第一次残差：加回原始输入。
        h = x + attn_update

        # 对 h 归一化，再计算 FFN 更新量。
        ffn_input = self.ln_2(h)
        ffn_update = self.ffn(ffn_input)

        # 第二次残差：加回第一次更新后的 h。
        y = h + ffn_update
        return y
