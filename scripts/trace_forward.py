"""追踪字符、整数编号与嵌入层前向计算的实际数值和 Shape。"""

import argparse

import torch

from mini_gpt.attention import CausalSelfAttention
from mini_gpt.blocks import DecoderBlock, FeedForward
from mini_gpt.embeddings import TokenPositionEmbedding
from mini_gpt.losses import make_next_token_batch, next_token_loss
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


def show_tensor(name: str, value: torch.Tensor) -> None:
    """打印教学示例中的真实张量。"""
    print(f"\n{name}：shape={tuple(value.shape)}, dtype={value.dtype}")
    print(value)


def trace_embeddings(token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
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
        return output


def trace_single_head_attention(x: torch.Tensor) -> None:
    """追踪单头计算的实际数值，并验证未来位置不会影响前面的输出。"""
    B, T, C = x.shape
    layer = CausalSelfAttention(n_embd=C, n_head=1)
    print("\n单头因果自注意力：接收上面的 Embedding 输出。")
    print("演示层使用 Q=0.02X、K=0.01X、V=X，输出投影为恒等映射。")
    print("这些人为设置仅用于本次演示，Attention 类本身仍使用默认初始化。")

    with torch.no_grad():
        # 控制分数幅度，避免本例较大的嵌入值让权重过早趋近 0 或 1。
        identity = torch.eye(C, dtype=x.dtype, device=x.device)
        layer.q_proj.weight.copy_(0.02 * identity)
        layer.k_proj.weight.copy_(0.01 * identity)
        layer.v_proj.weight.copy_(identity)
        layer.out_proj.weight.copy_(identity)
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            projection.bias.zero_()

        # 分步展示用于理解；最后用真实 forward 调用验证整条计算链。
        q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
        scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(masked_scores, dim=-1)
        context = weights @ v
        output = layer(x)

        print(f"\nB={B}, T={T}, C={C}, H=1")
        for name, value in (
            ("Attention 输入 X [B,T,C]", x),
            ("查询 Q [B,T,C]", q),
            ("键 K [B,T,C]", k),
            ("值 V [B,T,C]", v),
            ("缩放分数 [B,T,T]", scores),
            ("允许关注的位置 [T,T]", mask),
            ("遮罩后的分数 [B,T,T]", masked_scores),
            ("注意力权重 [B,T,T]", weights),
            ("每行权重之和 [B,T]", weights.sum(dim=-1)),
            ("加权汇总 context [B,T,C]", context),
            ("实际 forward 输出 [B,T,C]", output),
        ):
            show_tensor(name, value)

        torch.testing.assert_close(output, layer.out_proj(context))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
        assert torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0
        print("\n核对通过：实际输出与分步计算一致，未来位置权重为 0，每行权重之和为 1。")

        # 改变最后一个位置，前 T-1 个位置的输出应保持一致。
        changed_x = x.clone()
        changed_x[:, -1, :] += 10.0
        changed_output = layer(changed_x)
        torch.testing.assert_close(output[:, :-1], changed_output[:, :-1])
        difference = (output[:, :-1] - changed_output[:, :-1]).abs().max().item()
        print(f"因果性核对：修改最后位置后，前 {T - 1} 个位置输出的最大差值 = {difference}")
        print("以上展示单头路径；双头的独立权重见双头演示。")


def trace_multi_head_attention() -> None:
    """用固定输入展示两个头如何对同一位置分配不同的注意力。"""
    tokenizer = CharTokenizer("abc")
    token_ids = torch.tensor([tokenizer.encode("abc")], dtype=torch.long, device="cpu")
    embedding = TokenPositionEmbedding(vocab_size=3, block_size=3, n_embd=8)
    layer = CausalSelfAttention(n_embd=8, n_head=2)
    print("\n双头数据流：文本 'abc'，B=1、T=3、C=8、H=2、D=4。")
    print("演示人为设置嵌入、恒等投影和零偏置，帮助观察数值；实际类仍使用默认初始化。")

    with torch.no_grad():
        # 两组特征刻意设置为不同模式，便于对比两张注意力矩阵。
        embedding.token_embedding.weight.copy_(torch.tensor([
            [2., 0., 0., 0., 0., 1., 0., 0.],
            [1., 0., 0., 0., 3., 0., 0., 0.],
            [0., 1., 0., 0., 2., 0., 0., 0.],
        ]))
        embedding.position_embedding.weight.zero_()
        for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj):
            projection.weight.copy_(torch.eye(8))
            projection.bias.zero_()

        x = embedding(token_ids)
        B, T, C = x.shape
        H, D = layer.n_head, layer.head_dim
        q_flat, k_flat, v_flat = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
        q = q_flat.reshape(B, T, H, D).transpose(1, 2)
        k = k_flat.reshape(B, T, H, D).transpose(1, 2)
        v = v_flat.reshape(B, T, H, D).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / D ** 0.5
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        weights = masked_scores.softmax(dim=-1)
        context = weights @ v
        merged = context.transpose(1, 2).reshape(B, T, C)
        output = layer(x)

        for name, value in (
            ("字符编号 [B,T]", token_ids),
            ("Embedding 输出 X [B,T,C]", x),
            ("投影后、拆头前的 Q [B,T,C]", q_flat),
            ("reshape 后的 Q [B,T,H,D]", q_flat.reshape(B, T, H, D)),
            ("交换维度后的 Q [B,H,T,D]", q),
            ("交换维度后的 K [B,H,T,D]", k),
            ("交换维度后的 V [B,H,T,D]", v),
            ("各头缩放分数 [B,H,T,T]", scores),
            ("各头遮罩后分数 [B,H,T,T]", masked_scores),
            ("各头注意力权重 [B,H,T,T]", weights),
            ("各头各行权重之和 [B,H,T]", weights.sum(dim=-1)),
            ("各头上下文 [B,H,T,D]", context),
            ("拼接后的上下文 [B,T,C]", merged),
            ("实际 forward 输出 [B,T,C]", output),
        ):
            show_tensor(name, value)

        for head_index in range(H):
            show_tensor(f"位置 1 在头 {head_index} 中的权重", weights[0, head_index, 1])
        print("同一位置的两个头可产生不同关注分配，未来位置的权重均为 0。")
        torch.testing.assert_close(output, layer.out_proj(merged))
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(B, H, T))
        assert torch.count_nonzero(torch.triu(weights, diagonal=1)).item() == 0

        changed_x = x.clone()
        changed_x[:, -1] += 10.0
        changed_output = layer(changed_x)
        torch.testing.assert_close(output[:, :-1], changed_output[:, :-1])
        difference = (output[:, :-1] - changed_output[:, :-1]).abs().max().item()
        print(f"双头因果性验证通过：修改未来位置后，前面位置的最大输出差值 = {difference}")
        print("核对通过：实际多头 forward 与分步计算一致。")


def trace_ffn() -> None:
    """用两维小输入展示升维、非线性激活、降维与逐位置独立计算。"""
    layer = FeedForward(n_embd=2)
    x = torch.tensor([[[-1., 2.], [0., 1.], [1., -2.]]], device="cpu")
    print("\nFFN 数据流：B=1、T=3、C=2，中间维度为 8。")
    print("使用固定演示权重和零偏置，输入模拟 Attention 输出；实际类仍使用默认初始化。")

    with torch.no_grad():
        layer.up_proj.weight.copy_(torch.tensor([
            [1., 0.], [0., 1.], [-1., 0.], [0., -1.],
            [1., 1.], [1., -1.], [-1., 1.], [0., 0.],
        ]))
        layer.up_proj.bias.zero_()
        layer.down_proj.weight.copy_(torch.tensor([
            [1., 0., 0., 0., 0.5, 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.5, 0., 0.],
        ]))
        layer.down_proj.bias.zero_()

        expanded = layer.up_proj(x)
        activated = layer.act(expanded)
        restored = layer.down_proj(activated)
        output = layer(x)
        for name, value in (
            ("输入 X [B,T,C]", x),
            ("升维权重 [4C,C]", layer.up_proj.weight),
            ("升维结果 [B,T,4C]", expanded),
            ("GELU 激活结果 [B,T,4C]", activated),
            ("降维权重 [C,4C]", layer.down_proj.weight),
            ("降维结果 [B,T,C]", restored),
            ("实际 forward 输出 [B,T,C]", output),
        ):
            show_tensor(name, value)
        torch.testing.assert_close(output, restored)
        print("\n核对通过：实际 forward 与分步计算一致。GELU 改变数值，保留 Shape。")

        # 只改变中间位置，不改变前后两个位置的输入。
        changed_x = x.clone()
        changed_x[:, 1] += 10.0
        changed_output = layer(changed_x)
        torch.testing.assert_close(output[:, [0, 2]], changed_output[:, [0, 2]])
        difference = (output[:, [0, 2]] - changed_output[:, [0, 2]]).abs().max().item()
        show_tensor("修改位置 1 后的输出 [B,T,C]", changed_output)
        print(f"逐位置独立性验证通过：位置 0 和 2 的最大输出差值 = {difference}")
        print("这验证了 FFN 本身不在位置之间交换信息。")


def trace_decoder_block() -> None:
    """只查看当前 __init__ 已创建的模块及其参数形状。"""
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu")
    print("\nDecoderBlock.__init__ 构造结果：C=4、H=2。")
    print("以下按创建顺序展示四个子模块及其实际参数形状。")

    for index, (name, module) in enumerate(block.named_children(), start=1):
        print(f"\n{index}. self.{name}：")
        print(module)
        for parameter_name, parameter in module.named_parameters():
            print(f"  {parameter_name}：shape={tuple(parameter.shape)}, device={parameter.device}")


def trace_decoder_block_forward() -> None:
    """记录已实现的 forward 实际经过的子模块输入输出。"""
    torch.manual_seed(42)
    block = DecoderBlock(n_embd=4, n_head=2).to("cpu").eval()
    x = torch.tensor([[[1., 2., 3., 4.], [2., 0., 1., -1.], [-1., 1., 0., 2.]]],
                     device="cpu")
    print("\nDecoderBlock.forward 实际数据流：B=1、T=3、C=4、H=2。")
    print("输入为教学小张量；固定随机种子 42，权重使用类的默认初始化。")
    values = {}

    def record(name):
        # 只记录真实调用，不在追踪脚本中另行串联一份前向计算。
        def hook(module, inputs, output):
            values[name] = (inputs[0].detach().clone(), output.detach().clone())
        return hook

    handles = [module.register_forward_hook(record(name))
               for name, module in block.named_children()]
    try:
        with torch.no_grad():
            output = block(x)
    finally:
        for handle in handles:
            handle.remove()

    # ln_2 实际收到的输入就是学习者在 forward 中算出的第一次残差 h。
    h = values["ln_2"][0]
    for name, value in (
        ("1. 原始输入 X [B,T,C]", x),
        ("2. LN1(X) [B,T,C]", values["ln_1"][1]),
        ("3. Attention 更新量 [B,T,C]", values["attn"][1]),
        ("4. 第一次残差 h [B,T,C]", h),
        ("5. LN2(h) [B,T,C]", values["ln_2"][1]),
        ("6. FFN 更新量 [B,T,C]", values["ffn"][1]),
        ("7. 完整 Block 实际输出 Y [B,T,C]", output),
    ):
        show_tensor(name, value)
    torch.testing.assert_close(h, x + values["attn"][1])
    torch.testing.assert_close(output, h + values["ffn"][1])
    print("\n核对通过：实际 forward 中，h = X + Attention 更新量，Y = h + FFN 更新量。")


def trace_gpt_forward() -> None:
    """只记录实际 MiniGPT 前向调用，贯通字符编号、嵌入、各层与词表分数。"""
    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    text = "cab a"
    token_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device="cpu")
    model = MiniGPT(tokenizer.vocab_size, 8, 8, 2, 2).to("cpu").eval()
    print(f"\n最小 GPT 实际前向：输入 {text!r}，词表 {tokenizer.stoi}。")
    print("B=1、T=5、C=8、H=2、D=4、V=4，堆叠 2 个 Decoder Block。")
    print("固定随机种子 42，使用默认初始化的未训练参数。")
    values = {}

    def record(name):
        def hook(module, inputs, output):
            values[name] = output.detach().clone()
        return hook

    layers = [
        ("字符查表结果 [B,T,C]", model.embedding.token_embedding),
        ("位置查表结果 [T,C]", model.embedding.position_embedding),
        ("Embedding 相加结果 [B,T,C]", model.embedding),
    ]
    layers.extend((f"第 {index} 个 Block 输出 [B,T,C]", block)
                  for index, block in enumerate(model.blocks))
    layers.extend([
        ("最终 LayerNorm 输出 [B,T,C]", model.ln_f),
        ("LM Head 输出 logits [B,T,V]", model.lm_head),
    ])
    handles = [module.register_forward_hook(record(name)) for name, module in layers]
    try:
        with torch.no_grad():
            logits = model(token_ids)
    finally:
        for handle in handles:
            handle.remove()

    show_tensor("字符编号 [B,T]", token_ids)
    for name, module in layers:
        show_tensor(name, values[name])
    torch.testing.assert_close(logits, values["LM Head 输出 logits [B,T,V]"])
    print("\n最后一个输入位置对应的下一字符分数：")
    for token_id, score in enumerate(logits[0, -1]):
        print(f"  字符 {tokenizer.itos[token_id]!r}，编号 {token_id}，logit = {score.item():.6f}")
    print("以上是实际前向计算的原始分数，模型尚未训练。")


def trace_next_token_loss() -> None:
    """展示已完成的移位切片、真实模型分数和逐位置交叉熵。"""
    torch.manual_seed(42)
    tokenizer = CharTokenizer("abc ")
    text = "cab a"
    token_ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device="cpu")
    inputs, targets = make_next_token_batch(token_ids)
    model = MiniGPT(tokenizer.vocab_size, 8, 8, 2, 2).to("cpu").eval()
    print(f"\n下一字符损失数据流：完整文本 {text!r}，词表 {tokenizer.stoi}。")
    print("完整长度 L=5；移位后的 B=1、T=4、C=8、H=2、V=4。")
    print("固定随机种子 42，使用默认初始化的未训练参数。")

    with torch.no_grad():
        logits = model(inputs)
        loss = next_token_loss(logits, targets)
        # 概率只用于展示；实际损失函数接收上面的原始 logits。
        probabilities = logits.softmax(dim=-1)
        correct_probabilities = probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        # 用稳定的 log_softmax 展示每个位置的负对数概率。
        position_losses = -logits.log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    for name, value in (
        ("1. 完整编号 [B,L]", token_ids),
        ("2. 模型输入 input_ids [B,T]", inputs),
        ("3. 对齐后的正确标签 targets [B,T]", targets),
        ("4. 模型原始输出 logits [B,T,V]", logits),
        ("5. 展平分数 [B*T,V]", logits.reshape(-1, tokenizer.vocab_size)),
        ("6. 展平标签 [B*T]", targets.reshape(-1)),
        ("7. 各字符概率 [B,T,V]", probabilities),
        ("8. 正确字符的概率 [B,T]", correct_probabilities),
        ("9. 每个位置的 loss [B,T]", position_losses),
        ("10. next_token_loss 返回的平均 loss，标量", loss),
    ):
        show_tensor(name, value)

    print("\n按位置对齐：loss = -log(正确字符的概率)")
    for position in range(inputs.size(1)):
        target_char = tokenizer.itos[targets[0, position].item()]
        print(f"  位置 {position}：上下文 {text[:position + 1]!r} → 标签 {target_char!r}，"
              f"概率 {correct_probabilities[0, position].item():.6f}，"
              f"loss {position_losses[0, position].item():.6f}")
    torch.testing.assert_close(loss, position_losses.mean())
    print(f"\n核对通过：各位置 loss 的平均值 = next_token_loss = {loss.item():.6f}。")
    print("本次只计算分数与损失，模型参数未更新。")


def main() -> None:
    parser = argparse.ArgumentParser(description="观察已实现模块的数据流")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--multi-head-only", action="store_true", help="只打印双头数据流")
    mode.add_argument("--ffn-only", action="store_true", help="只打印 FFN 数据流")
    mode.add_argument("--block-only", action="store_true", help="只查看 DecoderBlock 初始化的模块与参数形状")
    mode.add_argument("--block-forward-only", action="store_true", help="只追踪已实现 DecoderBlock 的实际前向数据流")
    mode.add_argument("--gpt-only", action="store_true", help="只追踪字符编号到 GPT 词表分数的实际前向数据流")
    mode.add_argument("--loss-only", action="store_true", help="只追踪移位标签、逐位置交叉熵与平均损失")
    args = parser.parse_args()
    if args.loss_only:
        trace_next_token_loss()
        return
    if args.gpt_only:
        trace_gpt_forward()
        return
    if args.block_forward_only:
        trace_decoder_block_forward()
        return
    if args.block_only:
        trace_decoder_block()
        return
    if args.ffn_only:
        trace_ffn()
        return
    if args.multi_head_only:
        trace_multi_head_attention()
        return

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

    embedded = trace_embeddings(x, tokenizer.vocab_size)
    trace_single_head_attention(embedded)
    trace_multi_head_attention()
    trace_ffn()


if __name__ == "__main__":
    main()
