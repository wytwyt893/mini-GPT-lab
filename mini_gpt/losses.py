"""第 7.4 节练习：移位标签与下一字符预测损失。"""

import torch
from torch.nn import functional as F


def make_next_token_batch(token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将完整编号 [B,L] 切成输入与标签 [B,L-1]，其中 L 至少为 2。

    每个标签是对应输入位置的下一个字符；编号张量使用 torch.long。
    """
    if token_ids.ndim != 2:
        raise ValueError("完整编号张量必须为二维 [B,L]")
    if token_ids.size(1) < 2:
        raise ValueError("构造下一字符标签至少需要两个字符")

    # 保留批量维度，去掉每条序列的最后一个编号。
    # 举例：token_ids = [[1, 2, 3, 4]] → input_ids = [[1, 2, 3]]
    # 目的是让每个位置的标签是对应输入位置的下一个字符。
    input_ids = token_ids[:, :-1]

    # 保留批量维度，去掉每条序列的第一个编号。
    # 举例：token_ids = [[1, 2, 3, 4]] → targets = [[2, 3, 4]]
    targets = token_ids[:, 1:]

    return input_ids, targets


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """输入原始分数 [B,T,V] 和已对齐的标签 [B,T]，返回平均交叉熵标量。

    targets 使用 torch.long；输入与标签已经移位对齐，此处直接按位置比较。
    """
    B, T, V = logits.shape

    # 合并批量和位置维，保留词表维。
    flat_logits = logits.reshape(B * T, V)

    # 按同样的顺序展开正确答案。
    flat_targets = targets.reshape(B * T)

    return F.cross_entropy(flat_logits, flat_targets)
