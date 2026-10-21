"""追踪字符、整数编号和 CPU 张量之间的数据转换及真实 Shape。"""

import torch

from mini_gpt.tokenizer import CharTokenizer


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


if __name__ == "__main__":
    main()
