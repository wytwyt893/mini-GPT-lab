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

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """无缓存贪心生成：输入 [B,T]，返回含原始输入的 [B,T+max_new_tokens]。

        每轮选择最大概率对应的编号；调用前由外部设置 model.eval()。
        """
        if token_ids.ndim != 2 or token_ids.size(0) == 0 or token_ids.size(1) == 0:
            raise ValueError("生成输入必须为非空二维编号张量 [B,T]")
        if max_new_tokens < 0:
            raise ValueError("新增 token 数量不能为负数")

        for _ in range(max_new_tokens):
            # 保留最近的上下文，避免超过位置 Embedding 的长度限制。
            context = token_ids[:, -self.block_size:]

            # 复用当前模型进行前向计算。
            logits = self(context)

            # 只取最后一个位置，用它预测接下来的字符。
            last_logits = logits[:, -1, :]

            # 转成下一字符的概率分布。
            probs = torch.softmax(last_logits, dim=-1)

            # 选择概率最大的编号，保持 [B,1]，便于拼接。
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # 接到完整序列末尾，下一轮就能看到自己刚生成的字符。
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids
