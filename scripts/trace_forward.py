"""追踪字符、整数编号与嵌入层前向计算的实际数值和 Shape。"""

import torch

from mini_gpt.embeddings import TokenPositionEmbedding
from mini_gpt.tokenizer import CharTokenizer


def show_tensor(name: str, value: torch.Tensor) -> None:
    """打印教学示例中的真实张量。"""
    print(f"\n{name}：shape={tuple(value.shape)}, dtype={value.dtype}")
    print(value)


def trace_embeddings(token_ids: torch.Tensor, vocab_size: int) -> None:
    """用便于手算的小表拆解前向过程，并与实际 forward 输出核对。"""
    layer = TokenPositionEmbedding(vocab_size, block_size=8, n_embd=3)
    print("\n嵌入层数据流：演示使用人为设置的小表，方便逐项核对。")
    print("这些值只设置在本次新建的演示层中，正常嵌入层仍使用默认初始化。")

    with torch.no_grad():
        # 字符表每一行依次为 [1,2,3]、[4,5,6] 等。
        layer.token_embedding.weight.copy_(
            torch.arange(1, vocab_size * 3 + 1, dtype=torch.float32).reshape(vocab_size, 3)
        )
        # 第 t 行为 [10t,10t,10t]，让位置相加的结果容易手算。
        layer.position_embedding.weight.copy_(
            torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 3) * 10
        )

        B, T = token_ids.shape
        print(f"\nB={B}, T={T}, C=3")
        show_tensor("输入 token_ids [B,T]", token_ids)
        show_tensor("字符嵌入权重 [V,C]", layer.token_embedding.weight)
        show_tensor("位置嵌入权重 [block_size,C]", layer.position_embedding.weight)

        # 拆开计算以展示中间值，最终由学习者实现的 forward 给出实际输出。
        token_vectors = layer.token_embedding(token_ids)
        position_ids = torch.arange(T, dtype=torch.long, device=token_ids.device)
        position_vectors = layer.position_embedding(position_ids)
        output = layer(token_ids)

        show_tensor("字符查表结果 token_vectors [B,T,C]", token_vectors)
        show_tensor("当前位置编号 position_ids [T]", position_ids)
        show_tensor("位置查表结果 position_vectors [T,C]", position_vectors)
        show_tensor("实际 forward 输出 [B,T,C]", output)
        torch.testing.assert_close(output, token_vectors + position_vectors)
        print("\n核对通过：forward 输出 = 字符向量 + 广播后的位置向量。")


def main() -> None:
    tokenizer = CharTokenizer("abc ") # 调用 mini_gpt\tokenizer.py
    token_ids = tokenizer.encode("cab a") # 转换为整数编号
    print("字符编号：", token_ids)

    # 将 Python 列表转换为整数张量。
    # torch.long 表示 64 位整数类型，device="cpu" 表示在 CPU 上创建张量
    x = torch.tensor(token_ids, dtype=torch.long, device="cpu") # 把列表转换为张量
    print("单条序列 Shape：", tuple(x.shape))

    # 在最前面增加批量维度。
    x = x.unsqueeze(0) # unsqueeze(0) 在第 0 维增加一个维度，表示批量大小为 1
    print("带批量维度的 Shape：", tuple(x.shape))

    # tolist() 将张量转换为 Python 列表，decode() 将编号转换回字符
    # 还原文本，repr() 用于显示字符串的原始表示形式，包括转义字符
    text = tokenizer.decode(x[0].tolist())
    print("还原文本：", repr(text))

    trace_embeddings(x, tokenizer.vocab_size)


if __name__ == "__main__":
    main()
