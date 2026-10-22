"""第 7 章练习：组合已有模块，建立最小 GPT 与语言模型输出头。"""

import torch
from torch import nn

from mini_gpt.blocks import DecoderBlock
from mini_gpt.embeddings import TokenPositionEmbedding


class MiniGPT(nn.Module):
    """输入字符编号 [B,T]，输出下一字符的词表分数 [B,T,V]。

    参数：
        vocab_size：词表大小 V。
        block_size：允许的最大序列长度。
        n_embd：隐藏维度 C。
        n_head：每个 Decoder Block 内的注意力头数 H。
        n_layer：堆叠的 Decoder Block 数量，与头数不同。
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
    ) -> None:
        super().__init__()
        if n_layer <= 0:
            raise ValueError("解码器层数必须为正数")
        self.vocab_size = vocab_size
        self.block_size = block_size

        # 建立 self.embedding，复用 TokenPositionEmbedding。
        self.embedding = TokenPositionEmbedding(vocab_size, block_size, n_embd)

        # 建立 self.blocks，用 nn.ModuleList 保存 n_layer 个独立的 DecoderBlock。
        self.blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head)
            for _ in range(n_layer)
        ])

        # 建立最终归一化 self.ln_f，归一化隐藏维度 C。
        self.ln_f = nn.LayerNorm(n_embd)

        # 建立 self.lm_head，用无偏置线性层将 C 维映射为 V 个分数。
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """输入 torch.long 编号 [B,T]，返回浮点 logits [B,T,V]。"""
        # 字符编号变成带位置信息的向量：[B,T] → [B,T,C]。
        x = self.embedding(token_ids)

        # 依次经过每个 Decoder Block，形状始终为 [B,T,C]。
        for block in self.blocks:
            x = block(x)

        # 对最后一层输出进行归一化：[B,T,C]。
        x = self.ln_f(x)

        # 转成词表中各字符的分数：[B,T,C] → [B,T,V]。
        logits = self.lm_head(x)
        return logits
