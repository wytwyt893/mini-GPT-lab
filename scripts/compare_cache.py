"""记录实际缓存计算，比较 Attention 或整个 GPT 的生成结果。"""

import argparse

import torch

from mini_gpt.attention import CausalSelfAttention
from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


@torch.no_grad()
def trace_attention_cache() -> None:
    """记录真实 Attention 调用，比较整段与分段输出，不调用未完成的模型缓存。"""
    torch.manual_seed(42)
    layer = CausalSelfAttention(n_embd=8, n_head=2).to("cpu").eval()
    x = torch.randn(1, 6, 8, device="cpu")
    full_output = layer(x)
    print("本模式仅验证 Attention 层；整模型生成对比请不带参数运行。")
    print("CPU、固定随机种子 42，输入为模拟 Embedding 输出的随机隐藏向量。")
    print(f"整段输入：{tuple(x.shape)}，整段输出：{tuple(full_output.shape)}")

    projection_lengths = {name: [] for name in ("q_proj", "k_proj", "v_proj")}

    def record_input_length(name):
        def hook(module, args, output):
            projection_lengths[name].append(args[0].size(1))
        return hook

    handles = [getattr(layer, name).register_forward_hook(record_input_length(name))
               for name in projection_lengths]
    cache = None
    outputs = []
    start = 0
    try:
        for step, length in enumerate((3, 1, 2), start=1):
            current = x[:, start:start + length]
            past_length = 0 if cache is None else cache[0].size(2)
            output, cache = layer(current, past_kv=cache, use_cache=True)
            print(f"\n第 {step} 次调用：历史 P={past_length}，本次 T={length}")
            print(f"  当前输入 [B,T,C]：{tuple(current.shape)}")
            print(f"  当前输出 [B,T,C]：{tuple(output.shape)}")
            print(f"  累积 K/V [B,H,P+T,D]：{tuple(cache[0].shape)} / {tuple(cache[1].shape)}")
            outputs.append(output)
            start += length
    finally:
        for handle in handles:
            handle.remove()

    cached_output = torch.cat(outputs, dim=1)
    max_difference = (full_output - cached_output).abs().max().item()
    print(f"\n拼接各次当前输出：{tuple(cached_output.shape)}")
    print(f"与整段 Attention 输出的最大绝对误差：{max_difference:.10f}")
    for name, lengths in projection_lengths.items():
        print(f"真实 {name} 调用处理的序列长度：{lengths}")
    torch.testing.assert_close(cached_output, full_output, rtol=1e-5, atol=1e-6)
    print("验证通过：历史 K/V 复用；本次只投影新增位置；输出在浮点容差内一致。")


@torch.no_grad()
def trace_model_cache() -> None:
    """用前向钩子记录两条真实生成路径，逐步核对最后位置的 logits。"""
    torch.manual_seed(42)
    model = MiniGPT(4, 8, 8, 2, 2).to("cpu").eval()
    tokenizer = CharTokenizer("abc ")
    token_ids = torch.tensor([tokenizer.encode("cab")], dtype=torch.long, device="cpu")

    def observe_generation(generate):
        records = []

        def record(module, args, output):
            if isinstance(output, tuple):
                logits, cache = output
                cache_shapes = [(tuple(k.shape), tuple(v.shape)) for k, v in cache]
            else:
                logits, cache_shapes = output, None
            records.append({
                "input_shape": tuple(args[0].shape),
                "logits_shape": tuple(logits.shape),
                "last_logits": logits[:, -1, :].clone(),
                "cache_shapes": cache_shapes,
            })

        handle = model.register_forward_hook(record)
        try:
            generated = generate(token_ids, max_new_tokens=7)
        finally:
            handle.remove()
        return generated, records

    plain, plain_records = observe_generation(model.generate)
    cached, cached_records = observe_generation(model.generate_cached)
    assert torch.equal(plain, cached), "带缓存与无缓存生成的编号不同"
    assert len(plain_records) == len(cached_records) == 7

    print("CPU、固定随机种子 42、默认初始化的未训练模型；只验证缓存一致性。")
    print("B=1、C=8、H=2、D=4、V=4，2 个 Block，block_size=8。")
    print("从 'cab' 生成 7 个字符，最后一轮包含窗口滑动后的缓存重建。")
    max_difference = 0.0
    for step, (plain_step, cached_step) in enumerate(zip(plain_records, cached_records), start=1):
        torch.testing.assert_close(cached_step["last_logits"], plain_step["last_logits"],
                                   rtol=1e-5, atol=1e-6)
        difference = (cached_step["last_logits"] - plain_step["last_logits"]).abs().max().item()
        max_difference = max(max_difference, difference)
        print(f"\n第 {step} 轮：无缓存输入 {plain_step['input_shape']}，缓存输入 {cached_step['input_shape']}")
        print(f"  本轮缓存路径 logits [B,T,V]：{cached_step['logits_shape']}")
        for layer_index, (k_shape, v_shape) in enumerate(cached_step["cache_shapes"]):
            print(f"  第 {layer_index} 层 K/V [B,H,P+T,D]：{k_shape} / {v_shape}")

    print(f"\n无缓存真实输入长度：{[r['input_shape'][1] for r in plain_records]}")
    print(f"缓存真实输入长度：{[r['input_shape'][1] for r in cached_records]}")
    print(f"逐轮最后位置 logits 的最大绝对误差：{max_difference:.10f}")
    print(f"无缓存生成：{tokenizer.decode(plain[0].tolist())!r}")
    print(f"缓存生成：{tokenizer.decode(cached[0].tolist())!r}")
    print("验证通过：各轮 logits 在浮点容差内一致，生成编号完全一致。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-only", action="store_true", help="只验证已完成的 Attention 缓存")
    args = parser.parse_args()
    if args.attention_only:
        trace_attention_cache()
    else:
        trace_model_cache()


if __name__ == "__main__":
    main()
