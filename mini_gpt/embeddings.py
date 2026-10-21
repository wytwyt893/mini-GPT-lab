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

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入为 torch.long 编号张量 [B, T]，输出为浮点张量 [B, T, C]。
           token_ids：字符编号张量，形状为 [B, T]，数据类型为 torch.long
        """
        # 取得序列长度 T，并检查是否超过 block_size。
        B, T = token_ids.shape
        if T > self.block_size:
            raise ValueError("输入序列超过最大上下文长度")

        # 查字符编号嵌入表，取得 [B, T, C]。
        token_vectors = self.token_embedding(token_ids)

        # 为每个位置构造编号，再查位置嵌入表。
        position_ids = torch.arange(T, dtype=torch.long, device=token_ids.device)
        position_vectors = self.position_embedding(position_ids)

        # 将字符向量与位置向量相加。
        return token_vectors + position_vectors
