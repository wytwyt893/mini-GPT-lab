"""第 8.1 节运行入口：调用学习者填写的无缓存贪心生成。"""

import argparse

import torch

from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


def main() -> None:
    """用固定小词表和未训练模型验证字符生成流程。"""
    parser = argparse.ArgumentParser(description="使用未训练的微型 GPT 查看贪心生成结果")
    parser.add_argument("--prompt", default="cab a", help="输入前缀，字符来自词表 'abc '")
    parser.add_argument("--max-new-tokens", type=int, default=6, help="新增字符数")
    parser.add_argument("--trace", action="store_true", help="打印每轮实际上下文、概率与选中的字符")
    args = parser.parse_args()

    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    token_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device="cpu")
    model = MiniGPT(tokenizer.vocab_size, 8, 8, 2, 2).to("cpu").eval()
    print(f"输入文本：{args.prompt!r}")
    print(f"输入编号：{token_ids.tolist()}，Shape={tuple(token_ids.shape)}")
    print("使用固定随机种子 42 和默认初始化的未训练模型。")

    steps = []

    def record_step(module, inputs, logits):
        # 记录 generate 内部真正调用的 forward，选择结果则从最终返回序列中读取。
        steps.append((inputs[0].detach().clone(), logits[:, -1, :].softmax(-1).detach().clone()))

    handle = model.register_forward_hook(record_step) if args.trace else None
    try:
        output = model.generate(token_ids, max_new_tokens=args.max_new_tokens)
    finally:
        if handle is not None:
            handle.remove()

    if args.trace:
        print("\n逐轮实际生成，概率顺序为 a、b、c、空格：")
        for index, (context, probabilities) in enumerate(steps):
            context_text = tokenizer.decode(context[0].tolist())
            next_id = output[0, token_ids.size(1) + index].item()
            probability_text = ", ".join(f"{p:.4f}" for p in probabilities[0].tolist())
            print(f"第 {index + 1} 轮：上下文 {context_text!r}，Shape={tuple(context.shape)}；"
                  f"概率 [{probability_text}] → 编号 {next_id}，字符 {tokenizer.itos[next_id]!r}")

    new_ids = output[0, token_ids.size(1):].tolist()
    print(f"新增编号：{new_ids}")
    print(f"新增文本：{tokenizer.decode(new_ids)!r}")
    print(f"完整文本：{tokenizer.decode(output[0].tolist())!r}")
    print(f"输出 Shape：{tuple(output.shape)}")


if __name__ == "__main__":
    main()
