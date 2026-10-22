"""追踪已完成的 Attention 缓存；整模型缓存生成对比尚未接通。"""

import argparse

import torch

from mini_gpt.attention import CausalSelfAttention


@torch.no_grad()
def trace_attention_cache() -> None:
    """记录真实 Attention 调用，比较整段与分段输出，不调用未完成的模型缓存。"""
    torch.manual_seed(42)
    layer = CausalSelfAttention(n_embd=8, n_head=2).to("cpu").eval()
    x = torch.randn(1, 6, 8, device="cpu")
    full_output = layer(x)
    print("当前仅验证 Attention 层，尚未接通整个 GPT 的缓存生成。")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-only", action="store_true", help="只验证已完成的 Attention 缓存")
    args = parser.parse_args()
    if args.attention_only:
        trace_attention_cache()
    else:
        print("整模型缓存生成尚未接通；使用 --attention-only 查看已完成部分的真实数据流。")


if __name__ == "__main__":
    main()
