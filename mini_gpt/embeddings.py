"""第 2 阶段练习：字符编号嵌入与学习式位置嵌入。"""

import torch
from torch import nn


class TokenPositionEmbedding(nn.Module):
    """将整数编号 [B, T] 转换为包含位置信息的向量 [B, T, C]
    其中：
        B：批量大小
        T：序列长度
        C：嵌入维度
    参数：
        vocab_size：词表大小 V
        block_size：允许的最大序列长度，当前输入长度 T 可以小于它
        n_embd：嵌入维度 C，即每个字符用多少个浮点数表示
    """
    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__() # super() 调用父类 nn.Module 的初始化方法
        self.block_size = block_size

        # 建立字符编号嵌入表 self.token_embedding，形状为 [V, C]。
        # nn.Embedding 是 PyTorch 提供的嵌入层，用于将整数编号映射为向量。
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 建立位置嵌入表 self.position_embedding，形状为 [block_size, C]。
        self.position_embedding = nn.Embedding(block_size, n_embd)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """输入 [B,T]，返回 [B,T,C]；位置编号从历史长度开始。"""
        B, T = token_ids.shape

        if position_offset < 0:
            raise ValueError("位置偏移不能为负数")
        if position_offset + T > self.block_size:
            raise ValueError("输入序列超过最大上下文长度")

        # 字符编号仍然只查询本次输入。
        token_vectors = self.token_embedding(token_ids)

        # 位置编号从 P 开始，依次为 P、P+1、……、P+T-1。
        position_ids = torch.arange(
            position_offset,
            position_offset + T,
            dtype=torch.long,
            device=token_ids.device,
        )
        position_vectors = self.position_embedding(position_ids)

        return token_vectors + position_vectors
