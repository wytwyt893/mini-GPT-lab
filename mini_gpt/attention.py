"""第 4 章练习：逐步实现多头因果自注意力。"""

import torch
from torch import nn

from mini_gpt.cache import KVCache, append_kv_cache


class CausalSelfAttention(nn.Module):
    """手写多头因果自注意力，输入和输出均为 [B,T,C]。

    参数：
        n_embd：隐藏维度 C
        n_head：注意力头数 H
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        if n_embd <= 0 or n_head <= 0:
            raise ValueError("隐藏维度和头数必须为正数")
        if n_embd % n_head != 0:
            raise ValueError("隐藏维度必须能被头数整除")

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # 建立三个独立的线性投影 q_proj、k_proj、v_proj，均为 C -> C。
        """
            Q:Query，查询向量,当前位置要匹配什么信息
            K:Key，键向量,各位置提供什么匹配特征
            V:Value，值向量,匹配后传递什么内容
        """
        # nn.Linear 是 PyTorch 提供的线性层，用于实现全连接层。
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        # 输出投影在合并各头后使用，维度同样为 C -> C。

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        """当前输入 [B,T,C]；默认返回同形状输出。

        past_kv 保存历史 [B,H,P,D]；use_cache=True 时返回
        (当前输出, 更新后的 K/V)，新缓存长度为 P+T。
        """
        # 用三个线性层分别计算 Q、K、V。
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 拆头：分别将 q、k、v 从 [B,T,C] 变为 [B,H,T,D]。
        """
        参数：
            B: 批量大小
            T: 序列长度
            C: 隐藏维度
            H: 注意力头数
            D: 每个头的维度，D = C / H
        举例：
        假设输入 x 的形状为 [2, 3, 4]，即 B=2, T=3, C=4。
        假设有两个头 H=2，那么每个头的维度 D = C / H = 4 / 2 = 2。
        则 q、k、v 先被 reshape 为 [2, 3, 2, 2]，即 [B,T,H,D]。
        其中：
        - 第一个维度 2 表示批量大小 B=2
        - 第二个维度 3 表示序列长度 T=3
        - 第三个维度 2 表示注意力头数 H=2
        - 第四个维度 2 表示每个头的维度 D=2
        之后再交换位置维和头维，得到 [2,2,3,2]，即 [B,H,T,D]。
        交换位置维和头维是为了方便后续计算注意力分数和加权求和。
        """
        # 先 reshape 为 [B,T,H,D]，再交换位置维和头维；H*D=C。
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # P：历史缓存包含多少个位置。
        past_length = 0 if past_kv is None else past_kv[0].size(2)

        # K/V 包含历史和当前的位置，Q 仍然只保留当前的位置。
        k, v = append_kv_cache(k, v, past_kv)

        # 计算缩放后的 QK 转置乘积。
        """
        计算注意力分数：scores = QK^T / sqrt(d_k)，这里为矩阵乘法。
        (q.size(-1) ** 0.5) 对应于缩放因子，d_k 是每个头的维度。
        """
        scores = q @ k.transpose(-2, -1) # @表示矩阵乘法
        scores = scores / (q.size(-1) ** 0.5)

        # 遮住未来位置，再沿键所在的维度执行 Softmax。
        # True 表示允许关注，False 表示未来位置。
        """
        人为假设每一行的原始分数都是 [1,2,3]
        如果误将未来分数设成 0，会得到（实际代码不使用此方式）：
        [[1, 0, 0],
         [1, 2, 0],
         [1, 2, 3]]

        实际代码将未来分数改成负无穷：
        [[1, -inf, -inf],
         [1, 2, -inf],
         [1, 2, 3]]

        经过 softmax 后：
        [[1.0000, 0.0000, 0.0000],
         [0.2689, 0.7311, 0.0000],
         [0.0900, 0.2447, 0.6652]]
        """
        # 遮罩为 [T,P+T]，允许当前查询读取历史和自身之前的位置。
        # 当前查询对应的完整序列位置：P、P+1、……。
        query_positions = past_length + torch.arange(T, device=x.device)

        # 历史与当前全部键的位置：0、1、……。
        key_positions = torch.arange(k.size(2), device=x.device)

        # 允许关注：键的位置不超过当前查询的位置。
        mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        scores = scores.masked_fill(~mask, float("-inf")) # ~mask分数改成负无穷，表示未来位置
        weights = torch.softmax(scores, dim=-1)

        # 用注意力权重对 V 加权求和。
        """
        假设三个位置提供的 V 向量为：
        V₀ = [10, 0]
        V₁ = [0, 10]
        V₂ = [10,10]

        位置 0 的权重是 [1, 0, 0]，所以它的新向量为:
        1 * V₀ + 0 * V₁ + 0 * V₂ = [10, 0]

        位置 1 的权重是 [0.2689, 0.7311, 0]，所以它的新向量为:
        0.2689 * V₀ + 0.7311 * V₁ + 0 * V₂ = [2.689, 7.311]

        位置 2 的权重是 [0.0900, 0.2447, 0.6652]，所以它的新向量为:
        0.0900 * V₀ + 0.2447 * V₁ + 0.6652 * V₂ ≈ [7.5527, 9.0997]
        """
        context = weights @ v

        # 合并：多头时 context 为 [B,H,T,D]。
        # 先交换回 [B,T,H,D]，再 reshape 为 [B,T,C]，最后进行输出投影。
        """
        举例：
        假设有两个头，每个头的维度为 2，输入序列长度为 3。
        context 的形状为 [B,H,T,D] = [B,2,3,2]。
        交换后为 [B,T,H,D] = [B,3,2,2]。
        reshape 后为 [B,T,C] = [B,3,4]，因为 C = H * D = 2 * 2 = 4。
        """
        context = context.transpose(1, 2).reshape(B, T, C)
        output = self.out_proj(context)

        if use_cache:
            return output, (k, v)

        return output
