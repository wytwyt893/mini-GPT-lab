"""第 8.2 节练习：单步更新与短循环训练前后的生成对比。"""

import argparse

import torch

from mini_gpt.losses import make_next_token_batch, next_token_loss
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


def train_step(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """对一批 [B,T] 输入和标签更新一次参数，返回更新前算出的 loss 数值。"""
    model.train()

    # 清理上一次留下的梯度。
    optimizer.zero_grad(set_to_none=True)

    # 用当前参数预测，计算这批输入的损失。
    logits = model(input_ids)
    loss = next_token_loss(logits, targets)

    # 计算损失对各参数的梯度。
    loss.backward()

    # 根据梯度更新模型参数。
    optimizer.step()

    return loss.item()


def train_loop(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    num_steps: int,
) -> list[float]:
    """重复更新同一模型，记录每次更新前的 loss；输入与标签均为 [B,T]。"""
    if num_steps <= 0:
        raise ValueError("训练步数必须为正数")
    losses = []
    for step in range(1, num_steps + 1):
        # 在上一轮参数的基础上，再更新一次。
        loss_value = train_step(model, optimizer, input_ids, targets)

        # 保存本步的 loss，方便观察变化。
        losses.append(loss_value)

        # 打印第一步、每 20 步，以及最后一步。
        if step == 1 or step % 20 == 0 or step == num_steps:
            print(f"第 {step}/{num_steps} 步，loss={loss_value:.6f}")
    return losses


def run_single_step_demo() -> None:
    """仅更新一次参数，比较同一模型、同一批输入的前后结果。"""
    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    text = "abc abc a"
    token_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device="cpu")
    input_ids, targets = make_next_token_batch(token_ids)
    model = MiniGPT(tokenizer.vocab_size, 8, 8, 2, 2).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

    print("单步训练：CPU、固定随机种子 42，本次只更新一次参数。")
    print(f"输入文本：{tokenizer.decode(input_ids[0].tolist())!r}，Shape={tuple(input_ids.shape)}")
    print(f"正确标签：{tokenizer.decode(targets[0].tolist())!r}，Shape={tuple(targets.shape)}")
    model.eval()
    with torch.no_grad():
        before_loss = next_token_loss(model(input_ids), targets).item()
    before_weight = model.lm_head.weight[0, 0].item()
    print(f"更新前 loss：{before_loss:.6f}")
    print(f"更新前 lm_head.weight[0,0]：{before_weight:.6f}")

    step_loss = train_step(model, optimizer, input_ids, targets)

    model.eval()
    with torch.no_grad():
        after_loss = next_token_loss(model(input_ids), targets).item()
    after_weight = model.lm_head.weight[0, 0].item()
    print(f"train_step 返回的本次更新前 loss：{step_loss:.6f}")
    print(f"更新后重新计算的 loss：{after_loss:.6f}")
    print(f"更新后 lm_head.weight[0,0]：{after_weight:.6f}")


def run_training_demo(num_steps: int) -> None:
    """使用四个固定窗口训练同一模型，对比更新前后的 loss 和生成文本。"""
    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    corpus = "abc " * 3
    # 四个窗口分别从 a、b、c、空格开始，每个取 9 个字符，移位后得到 [4,8]。
    texts = [corpus[start:start + 9] for start in range(4)]
    token_ids = torch.tensor([tokenizer.encode(text) for text in texts], dtype=torch.long, device="cpu")
    input_ids, targets = make_next_token_batch(token_ids)
    model = MiniGPT(tokenizer.vocab_size, 8, 8, 2, 2).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    prompt_ids = torch.tensor([tokenizer.encode("a")], dtype=torch.long, device="cpu")

    print(f"短训练实验：CPU、固定随机种子 42，共 {num_steps} 步。")
    print(f"输入 Shape={tuple(input_ids.shape)}，标签 Shape={tuple(targets.shape)}")
    for index in range(input_ids.size(0)):
        print(f"样本 {index}：{tokenizer.decode(input_ids[index].tolist())!r} → "
              f"{tokenizer.decode(targets[index].tolist())!r}")

    model.eval()
    with torch.no_grad():
        before_loss = next_token_loss(model(input_ids), targets).item()
    before_text = tokenizer.decode(model.generate(prompt_ids, 16)[0].tolist())
    print(f"训练前 loss：{before_loss:.6f}")
    print(f"训练前从 'a' 生成：{before_text!r}")

    train_loop(model, optimizer, input_ids, targets, num_steps)

    model.eval()
    with torch.no_grad():
        after_loss = next_token_loss(model(input_ids), targets).item()
    after_text = tokenizer.decode(model.generate(prompt_ids, 16)[0].tolist())
    print(f"训练后重新计算的 loss：{after_loss:.6f}")
    print(f"训练后从 'a' 生成：{after_text!r}")


def main() -> None:
    """默认保留单步演示，指定 --steps 100 后进入短循环实验。"""
    parser = argparse.ArgumentParser(description="微型 GPT 的 CPU 单步更新与短训练练习")
    parser.add_argument("--steps", type=int, default=1, help="训练步数；1 为原单步演示，建议短循环使用 100")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("训练步数必须为正数")
    # 仅限制本进程的线程，减少微型矩阵运算的多线程调度开销。
    torch.set_num_threads(1)
    if args.steps == 1:
        run_single_step_demo()
    else:
        run_training_demo(args.steps)


if __name__ == "__main__":
    main()
