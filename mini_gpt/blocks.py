"""第 5 章练习：逐位置前馈网络；后续在此组合 Pre-LN 解码器块。"""

import torch
from torch import nn


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
