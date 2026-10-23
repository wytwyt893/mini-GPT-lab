"""从字符到字符的终端入门导览，采集已有模型真实执行的中间张量。"""

import argparse
from dataclasses import dataclass, field
import inspect
import sys
from typing import Any

import torch

from mini_gpt.model import MiniGPT
from mini_gpt.tokenizer import CharTokenizer


def copy_value(value: Any) -> Any:
    """保存当时的数据，避免后续赋值影响已经记录的过程。"""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(copy_value(item) for item in value)
    if isinstance(value, list):
        return [copy_value(item) for item in value]
    return value


@dataclass
class CallRecord:
    """一次真实函数调用中，同名变量依次出现过的值。"""

    values: dict[str, list[Any]] = field(default_factory=dict)
    result: Any = None
    previous: dict[str, Any] = field(default_factory=dict, repr=False)

    def observe(self, variables: dict[str, Any], names: tuple[str, ...]) -> None:
        for name in names:
            if name in variables and (
                name not in self.previous or variables[name] is not self.previous[name]
            ):
                value = variables[name]
                self.values.setdefault(name, []).append(copy_value(value))
                self.previous[name] = value

    def last(self, name: str) -> Any:
        return self.values[name][-1]


@dataclass
class RoundRecord:
    """一次 GPT 前向的输入、子模块输出与内部计算。"""

    input_ids: torch.Tensor
    calls: dict[str, CallRecord] = field(default_factory=dict)
    modules: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    def output(self, path: str) -> Any:
        return self.modules[path][1]


class ActualTrace:
    """临时记录已有 forward/generate 的局部变量与模块调用，不替换计算。"""

    def __init__(self, model: MiniGPT) -> None:
        self.model = model
        self.rounds: list[RoundRecord] = []
        self.generation = CallRecord()
        self.paths = {id(module): path for path, module in model.named_modules()}
        self.targets: dict[Any, tuple[str, ...]] = {}
        watched = {
            "embedding": ("token_vectors", "position_ids", "position_vectors"),
            "block": ("attn_input", "h", "ffn_input", "ffn_update", "y"),
            "attention": ("q", "k", "v", "past_length", "scores", "mask", "weights", "context"),
            "model": ("past_length",),
            "generation": ("token_ids", "last_logits", "probs", "next_token"),
        }
        self.targets[model.forward.__func__.__code__] = watched["model"]
        self.targets[model.embedding.forward.__func__.__code__] = watched["embedding"]
        for block in model.blocks:
            self.targets[block.forward.__func__.__code__] = watched["block"]
            self.targets[block.attn.forward.__func__.__code__] = watched["attention"]
        self.generation_codes = {
            inspect.unwrap(model.generate).__code__,
            inspect.unwrap(model.generate_cached).__code__,
        }
        for code in self.generation_codes:
            self.targets[code] = watched["generation"]

    def trace_call(self, frame, event, arg):
        if event != "call" or frame.f_code not in self.targets:
            return None
        owner = frame.f_locals.get("self")
        if id(owner) not in self.paths:
            return None
        names = self.targets[frame.f_code]
        path = self.paths[id(owner)]
        if frame.f_code in self.generation_codes:
            record = self.generation
        else:
            if owner is self.model:
                self.rounds.append(RoundRecord(copy_value(frame.f_locals["token_ids"])))
            record = CallRecord()
            self.rounds[-1].calls[path] = record

        def trace_line(current_frame, current_event, current_arg):
            if current_event in ("line", "return"):
                record.observe(current_frame.f_locals, names)
            if current_event == "return":
                record.result = copy_value(current_arg)
            return trace_line

        return trace_line

    def hook_for(self, path: str):
        def record(module, args, output):
            if self.rounds:
                self.rounds[-1].modules[path] = (copy_value(args[0]), copy_value(output))
        return record

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        self.handles = [module.register_forward_hook(self.hook_for(path))
                        for path, module in self.model.named_modules() if path]
        sys.settrace(self.trace_call)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.settrace(self.previous_trace)
        for handle in self.handles:
            handle.remove()


@dataclass
class FlowRun:
    """生成的实际结果，以及讲解这些结果所需的记录。"""

    model: MiniGPT
    tokenizer: CharTokenizer
    prompt: str
    single_ids: torch.Tensor
    input_ids: torch.Tensor
    output_ids: torch.Tensor
    use_cache: bool
    trace: ActualTrace


def make_demo_model() -> MiniGPT:
    """用微型配置缩小屏幕上的数字表，不改变已有 GPT 的实现。"""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return MiniGPT(4, 8, 4, 2, 1).to("cpu").eval()


@torch.no_grad()
def capture_flow(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool = False,
) -> FlowRun:
    """实际调用 generate 或 generate_cached，记录后检查数据是否相互对得上。"""
    if not prompt or max_new_tokens < 1:
        raise ValueError("请输入至少一个字符，并至少生成一个新字符")
    if model.training:
        raise ValueError("请先调用 model.eval()，再观察生成过程")
    single_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device="cpu")
    input_ids = single_ids.unsqueeze(0)
    trace = ActualTrace(model)
    generate = model.generate_cached if use_cache else model.generate
    with trace:
        output_ids = generate(input_ids, max_new_tokens)
    run = FlowRun(model, tokenizer, prompt, single_ids, input_ids, output_ids, use_cache, trace)
    validate_flow(run)
    return run


def validate_flow(run: FlowRun) -> None:
    """核对采集结果与真实返回值，防止演示与实际模型脱节。"""
    generation = run.trace.generation.values
    steps = run.output_ids.size(1) - run.input_ids.size(1)
    assert len(run.trace.rounds) == len(generation["next_token"]) == steps
    assert len(generation["token_ids"]) == steps + 1
    torch.testing.assert_close(generation["token_ids"][-1], run.output_ids)
    for index, record in enumerate(run.trace.rounds):
        logits_result = record.calls[""].result
        logits = logits_result[0] if run.use_cache else logits_result
        torch.testing.assert_close(logits, record.output("lm_head"))
        torch.testing.assert_close(generation["last_logits"][index], logits[:, -1, :])
        probabilities = generation["probs"][index]
        torch.testing.assert_close(probabilities, logits[:, -1, :].softmax(-1))
        torch.testing.assert_close(generation["next_token"][index], probabilities.argmax(-1, keepdim=True))
        torch.testing.assert_close(
            generation["token_ids"][index + 1],
            torch.cat((generation["token_ids"][index], generation["next_token"][index]), dim=1),
        )
        embedding = record.calls["embedding"]
        torch.testing.assert_close(record.output("embedding"),
                                   embedding.last("token_vectors") + embedding.last("position_vectors"))
        for layer_index in range(len(run.model.blocks)):
            path = f"blocks.{layer_index}"
            block = record.calls[path]
            attention = record.calls[path + ".attn"]
            update = record.output(path + ".attn")
            if run.use_cache:
                update = update[0]
            torch.testing.assert_close(block.last("h"), record.modules[path][0] + update)
            torch.testing.assert_close(block.last("y"), block.last("h") + block.last("ffn_update"))
            weights = attention.last("weights")
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
            blocked = ~attention.last("mask").expand_as(weights)
            assert torch.count_nonzero(weights.masked_select(blocked)) == 0


def character(char: str) -> str:
    return "空格" if char == " " else repr(char)


class Guide:
    """只负责中文排版，展示的数据来自上面记录的真实调用。"""

    def __init__(self) -> None:
        self.number = 0

    def station(self, title: str, file: str, module: str, explanation: str) -> None:
        self.number += 1
        print(f"\n{'=' * 72}\n第 {self.number:02d} 站 | {title}")
        print(f"负责文件：{file}\n负责模块：{module}\n白话解释：{explanation}")

    def tensor(self, name: str, value: torch.Tensor) -> None:
        print(f"\n{name} | Shape={tuple(value.shape)} | 类型={value.dtype}")
        print(value)

    def heads(self, name: str, value: torch.Tensor) -> None:
        print(f"\n{name} | 总 Shape={tuple(value.shape)} [B,H,查询位置,键位置]")
        for head in range(value.size(1)):
            print(f"头 {head}：行是当前查询位置，列是上下文的键位置；是否可以读取由遮罩决定。")
            print(value[0, head])


def describe_round(run: FlowRun, index: int, guide: Guide, detailed: bool) -> None:
    """按实际调用顺序讲解一轮，后续轮次可以只保留主要变化。"""
    record = run.trace.rounds[index]
    values = run.trace.generation.values
    tokenizer = run.tokenizer
    before = values["token_ids"][index]
    after = values["token_ids"][index + 1]
    current = tokenizer.decode(record.input_ids[0].tolist())
    past_length = record.calls[""].last("past_length")
    method = "MiniGPT.generate_cached" if run.use_cache else "MiniGPT.generate"
    guide.station(f"第 {index + 1} 轮：准备预测下一个字符", "mini_gpt/model.py", method,
                  "每轮先读取已有内容，再预测一个新字符。模型内部一直传递数字编号。")
    print(f"本轮开始时的完整文本：{tokenizer.decode(before[0].tolist())!r}")
    print(f"本次真正输入 GPT 的字符：{current!r}；历史缓存长度 P={past_length}")
    guide.tensor("实际输入编号 [B,T]", record.input_ids)
    if run.use_cache:
        if past_length:
            print("历史 K/V 已保存；本轮只输入上次新生成的字符。")
        else:
            print("本轮从当前窗口完整计算缓存（首次填充，或窗口滑动后重建）。")
    elif before.size(1) > run.model.block_size:
        print("文本超过窗口，本次只读取最后 8 个字符；最终输出仍保留完整原文。")
    if detailed:
        embedding = record.calls["embedding"]
        guide.station("把字符编号变成数字向量，再加上位置", "mini_gpt/embeddings.py",
                      "TokenPositionEmbedding.forward", "每个字符先领取一个数字向量，再加上它的位置向量。")
        guide.tensor("A. 字符查表结果 [B,T,C]", embedding.last("token_vectors"))
        guide.tensor("B. 位置编号 [T]，从历史长度 P 开始", embedding.last("position_ids"))
        guide.tensor("C. 位置查表结果 [T,C]", embedding.last("position_vectors"))
        guide.tensor("D. 相加后的真实输出 [B,T,C]", record.output("embedding"))
        print("位置向量会给这条文本的对应位置使用；字符编号与位置编号是两套编号。")
        first_id = record.input_ids[0, 0].item()
        first_position = embedding.last("position_ids")[0].item()
        print(f"例如第一个输入字符 {character(tokenizer.itos[first_id])}："
              f"字符表查第 {first_id} 行，位置表查第 {first_position} 行。")
        print("只看它的第一个数字："
              f"{embedding.last('token_vectors')[0, 0, 0].item():.4f} + "
              f"{embedding.last('position_vectors')[0, 0].item():.4f} = "
              f"{record.output('embedding')[0, 0, 0].item():.4f}")

        for layer_index, block_module in enumerate(run.model.blocks):
            path = f"blocks.{layer_index}"
            attention_path = path + ".attn"
            block = record.calls[path]
            attention = record.calls[attention_path]
            guide.station(f"进入第 {layer_index} 个 Decoder Block，先整理数值", "mini_gpt/blocks.py",
                          "DecoderBlock.ln_1", "LayerNorm 调整数值尺度；文字位置和向量长度都保持原样。")
            guide.tensor("送给 Attention 的 LN1 输出 [B,T,C]", block.last("attn_input"))

            guide.station("从同一份输入算出 Q、K、V，再拆成多个头", "mini_gpt/attention.py",
                          "CausalSelfAttention.q_proj / k_proj / v_proj",
                          "Q 用来发起匹配，K 用来被匹配，V 提供最终传递的内容；三个线性层各有自己的参数。")
            for name in ("q", "k", "v"):
                guide.tensor(f"{name.upper()} 线性层的真实输出 [B,T,C]", record.output(attention_path + f".{name}_proj"))
            guide.tensor("拆头后的 Q [B,H,T,D]", attention.last("q"))
            print(f"把 C={block_module.attn.n_embd} 个数字分给 H={block_module.attn.n_head} 个头，"
                  f"每头 D={block_module.attn.head_dim} 个数字；每个头都保留当前输入的全部位置。")
            guide.tensor("参与本轮计算的 K [B,H,P+T,D]", attention.last("k"))
            guide.tensor("参与本轮计算的 V [B,H,P+T,D]", attention.last("v"))
            if run.use_cache:
                print("缓存追加由 mini_gpt/cache.py :: append_kv_cache 负责；"
                      f"历史 {past_length} 个位置 + 当前 {record.input_ids.size(1)} 个位置。")

            guide.station("计算位置之间的关注比例", "mini_gpt/attention.py",
                          "CausalSelfAttention.forward",
                          "先让 Q 与 K 打分，再缩小分数、挡住未来位置，最后用 Softmax 变成分配比例。")
            key_ids = before[0, -run.model.block_size:].tolist()
            print("矩阵各列：" + "，".join(
                f"列 {i} = 位置 {i} 的 {character(tokenizer.itos[token_id])}"
                for i, token_id in enumerate(key_ids)
            ))
            print("矩阵各行：" + "，".join(
                f"行 {i} = 正在处理位置 {past_length + i} 的 {character(tokenizer.itos[token_id])}"
                for i, token_id in enumerate(record.input_ids[0].tolist())
            ))
            scores = attention.values["scores"]
            guide.heads("A. Q @ K转置：原始匹配分数", scores[0])
            guide.heads(f"B. 除以 sqrt(D)={block_module.attn.head_dim ** 0.5:.4f}", scores[1])
            guide.tensor("C. 因果遮罩：True 可以看，False 不可以看", attention.last("mask"))
            guide.heads("D. 遮住未来后的分数；-inf 让未来位置分到 0", scores[2])
            guide.heads("E. Softmax 后的注意力权重，每行之和为 1", attention.last("weights"))
            print("这里的比例分配给上下文位置，用于读取信息。稍后的词表概率才用于选择输出字符。")
            for head, row in enumerate(attention.last("weights")[0, :, -1]):
                print(f"头 {head} 处理本次最后一个位置时的真实分配：" + "，".join(
                    f"位置 {i} 的 {character(tokenizer.itos[token_id])} 占 {row[i].item():.1%}"
                    for i, token_id in enumerate(key_ids)
                ))

            guide.station("按关注比例汇总 V，再合并两个头", "mini_gpt/attention.py",
                          "CausalSelfAttention.forward / out_proj",
                          "把各位置提供的 V 按比例相加，各头分别得到内容，再拼接并通过输出线性层。")
            guide.tensor("A. weights @ V，各头的汇总结果 [B,H,T,D]", attention.values["context"][0])
            weights = attention.last("weights")[0, 0, -1]
            contents = attention.last("v")[0, 0, :, 0]
            print("例如头 0 的最后一个位置，只算第一个数字：")
            print(" + ".join(f"({weight.item():.4f} × {value.item():.4f})"
                             for weight, value in zip(weights, contents))
                  + f" = {attention.values['context'][0][0, 0, -1, 0].item():.4f}")
            guide.tensor("B. 合并各头 [B,T,C]", attention.values["context"][-1])
            attention_output = record.output(attention_path)
            if run.use_cache:
                attention_output = attention_output[0]
            guide.tensor("C. 输出投影得到 Attention 更新量 [B,T,C]", attention_output)

            guide.station("第一次残差：保留原来的表示，再加上新信息", "mini_gpt/blocks.py",
                          "DecoderBlock.forward：h = x + attn_update", "对应位置、对应维度逐项相加。")
            guide.tensor("原始输入 x", record.modules[path][0])
            guide.tensor("Attention 更新量", attention_output)
            guide.tensor("相加得到 h", block.last("h"))

            guide.station("第二次整理数值，再进入 FFN", "mini_gpt/blocks.py",
                          "DecoderBlock.ln_2 / FeedForward.forward",
                          "FFN 分别加工每个位置的向量：先扩展数字数量，经过 GELU，再缩回原来的长度。")
            guide.tensor("A. LN2 输出 [B,T,C]", block.last("ffn_input"))
            guide.tensor("B. up_proj 升维 [B,T,4C]", record.output(path + ".ffn.up_proj"))
            guide.tensor("C. GELU 激活 [B,T,4C]", record.output(path + ".ffn.act"))
            guide.tensor("D. down_proj 降维，得到 FFN 更新量 [B,T,C]", block.last("ffn_update"))
            print("FFN 的信息加工发生在每个位置内部；位置之间的信息交流由 Attention 完成。")

            guide.station("第二次残差：把 FFN 更新量加回 h", "mini_gpt/blocks.py",
                          "DecoderBlock.forward：y = h + ffn_update", "得到这一整个 Block 的输出。")
            guide.tensor("相加前的 h", block.last("h"))
            guide.tensor("相加后的 Block 输出 y [B,T,C]", block.last("y"))

        guide.station("最后一次整理数值，再给词表里的字符打分", "mini_gpt/model.py",
                      "MiniGPT.ln_f / lm_head", "把每个位置的 C 个特征，变成 V 个候选字符各自的分数。")
        guide.tensor("最终 LayerNorm 输出 [B,T,C]", record.output("ln_f"))
        guide.tensor("LM Head 原始分数 logits [B,T,V]", record.output("lm_head"))
        print("最后一维的分数顺序是：" + "、".join(character(tokenizer.itos[i]) for i in range(tokenizer.vocab_size)))
        print("每一行负责预测该位置后面的字符；分数允许为负数，还不是概率。")
        print("本例 C=4 是一个位置的特征数，V=4 是候选字符数；这两个数恰好相同，含义不同。")
        print("LM Head 用各候选自己的权重向量与当前位置的特征向量做点积，得到各候选的分数。")
    else:
        positions = record.calls["embedding"].last("position_ids")
        print(f"本轮实际位置编号：{positions.tolist()}")
        print("本轮仍实际经过：Embedding → 各 Decoder Block → 最终 LayerNorm → LM Head。")
        print("要展开后续每轮的内部数字，可加 --all-rounds。")

    guide.station("只取最后一个位置，决定接下来输出谁", "mini_gpt/model.py", method,
                  "最后一个位置已经读取本轮允许的全部上下文；把它的词表分数转成概率，选择概率最大的编号。")
    guide.tensor("实际选取的 last_logits [B,V]", values["last_logits"][index])
    probabilities = values["probs"][index]
    guide.tensor("实际 Softmax 概率 [B,V]", probabilities)
    next_id = values["next_token"][index].item()
    print("\n候选字符       编号       概率       长条越长，概率越大")
    for token_id in range(tokenizer.vocab_size):
        probability = probabilities[0, token_id].item()
        bar = "#" * round(probability * 40)
        selected = " <- 本轮实际选择" if token_id == next_id else ""
        print(f"{character(tokenizer.itos[token_id]):<10} {token_id:<6} {probability:>8.2%}  {bar}{selected}")
    guide.tensor("argmax 选出的 next_token [B,1]", values["next_token"][index])

    guide.station("把选出的编号接到末尾，为下一轮准备输入", "mini_gpt/model.py", method,
                  "torch.cat 沿文本长度方向追加一个编号。每轮只增加一个字符。")
    guide.tensor("追加前 [B,原长度]", before)
    guide.tensor("追加后 [B,原长度+1]", after)
    print(f"为方便阅读，调用 tokenizer.decode 后看到：{tokenizer.decode(after[0].tolist())!r}")
    if run.use_cache:
        _, cache = record.calls[""].result
        for layer_index, (k, v) in enumerate(cache):
            print(f"第 {layer_index} 层已保存的 K/V：{tuple(k.shape)} / {tuple(v.shape)}")
        print("缓存包括本轮已处理的位置；刚选出的新编号会在下一轮进入缓存。")


def print_flow(run: FlowRun, all_rounds: bool = False) -> None:
    """从原文到最终字符串，用可顺序阅读的终端说明展示实际数据流。"""
    print("字符旅行：从你输入的文字，一站一站走到模型输出的文字")
    print("本示例使用 CPU、固定随机种子 42、未训练的微型模型，用于看懂计算过程。")
    print("小词典固定为 a、b、c、空格；默认模型只有 1 个 Block、2 个头，每个字符用 4 个数字表示。")
    print("张量 tensor 可以先当成有形状的数字表；Shape 表示每一层装了多少个元素。")
    print("B=文本条数；T=本次输入字符数；C=每个字符的特征数；H=头数；D=每头特征数；V=词表大小。")
    print("例如 Shape=(1,3,4)：1 条文本 × 3 个位置 × 每个位置 4 个数字。")
    print("屏幕小数按显示精度四舍五入；下面的数字采集自实际执行的模型。")
    print("全程路线：字符 → 编号 → 向量 → Attention → FFN → 词表分数 → 概率 → 新编号 → 字符")
    guide = Guide()
    guide.station("认识输入与词表", "mini_gpt/tokenizer.py", "CharTokenizer.__init__",
                  "先约定每种字符的编号，就像给每个字符发一张号码牌。")
    print(f"你的输入：{run.prompt!r}")
    for char, token_id in run.tokenizer.stoi.items():
        print(f"  {character(char)} → 编号 {token_id}")
    guide.station("字符换成编号", "mini_gpt/tokenizer.py", "CharTokenizer.encode",
                  "逐字符查词表；得到的整数是编号，大小不表示字符的重要程度。")
    print(f"字符：{[character(char) for char in run.prompt]}")
    print(f"编号：{run.single_ids.tolist()}")
    guide.station("把编号装进 PyTorch 数字表，再增加批量维度", "scripts/baby_flow.py",
                  "torch.tensor / unsqueeze(0)", "模型支持同时读多条文本；即使只有一条，也保留外面这一层。")
    guide.tensor("先得到一维编号 [T]", run.single_ids)
    guide.tensor("再得到带批量维度的编号 [B,T]", run.input_ids)
    for index in range(len(run.trace.rounds)):
        describe_round(run, index, guide, detailed=index == 0 or all_rounds)
    guide.station("把最终编号翻译回字符", "mini_gpt/tokenizer.py", "CharTokenizer.decode",
                  "按反向词表逐个查回字符，再把字符拼成字符串。由教学脚本调用，模型生成时始终处理编号。")
    output = run.output_ids[0].tolist()
    new_ids = output[run.input_ids.size(1):]
    print(f"完整编号：{output}")
    print(f"新增编号：{new_ids}")
    print(f"新增字符：{run.tokenizer.decode(new_ids)!r}")
    print(f"最终完整文本：{run.tokenizer.decode(output)!r}")
    print("\n自检通过：嵌入相加、两次残差、注意力归一化、未来位置被遮住、字符回填与实际输出均已核对。")
    print("模型参数的学习由 scripts/train_tiny.py 和 mini_gpt/losses.py 负责；本次展示的是生成数据流。")


def main(argv: list[str] | None = None) -> None:
    """终端运行入口，默认完整讲解首轮，再简洁展示第二轮。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="cab", help="输入字符，只使用 a、b、c 和空格")
    parser.add_argument("--max-new-tokens", type=int, default=2, help="生成字符数，至少 1 个")
    parser.add_argument("--cache", action="store_true", help="调用实际缓存生成，观察历史 K/V 复用")
    parser.add_argument("--all-rounds", action="store_true", help="每轮都展开完整内部计算")
    args = parser.parse_args(argv)
    if not args.prompt or args.max_new_tokens < 1:
        parser.error("请输入至少一个字符，并至少生成一个新字符")
    tokenizer = CharTokenizer("abc ")
    unknown = set(args.prompt) - set(tokenizer.stoi)
    if unknown:
        parser.error(f"本示例词表只包含 a、b、c、空格；输入中有未知字符：{sorted(unknown)!r}")
    run = capture_flow(make_demo_model(), tokenizer, args.prompt, args.max_new_tokens, args.cache)
    print_flow(run, args.all_rounds)


if __name__ == "__main__":
    main()
