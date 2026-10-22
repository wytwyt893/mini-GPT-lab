# mini-gpt-lab

用 Python + PyTorch 从零实现最小 Decoder-only GPT 的两日学习项目。每次只推进一个模块，先理解并测试当前模块，再由学习者确认进入下一模块。

**当前状态：已完成 Tokenizer、Embedding、手写多头因果自注意力、FFN、Pre-LN Decoder Block、最小 GPT、损失、无缓存生成和单步 CPU 参数更新。** 当前共 74 项测试通过；多轮训练循环待学习者填写，KV Cache 尚未开始。

学习方式：导师在聊天中解释概念和小段代码，由学习者填写并验证；可在同一轮推进几个相关小节，不一次性替学习者实现完整模块。已有教学 Markdown 保留为历史材料，后续指导在聊天中进行，阶段结果简要记入学习记录。

## 目标与边界

最终贯通：字符 → 字符级 Tokenizer → Token/Position Embedding → Multi-Head Causal Self-Attention → FFN → Decoder Block → LM Head → 自回归生成 → KV Cache。

采用经典 GPT 风格：Pre-LN、GELU、学习式位置 Embedding；Attention 手写。仅在 Windows CPU 上使用微型数据与模型，不需要 CUDA、模型权重或大型训练框架。

- 项目绝对路径：`C:\Users\86185\Desktop\agent\mini-gpt-lab`
- Git：独立学习仓库，`origin` 为个人仓库 [wytwyt893/mini-GPT-lab](https://github.com/wytwyt893/mini-GPT-lab)，使用 `main` 分支。
- 参考：[karpathy/minGPT](https://github.com/karpathy/minGPT)，位于 `reference/minGPT/`，只读、不安装、不修改源码，不复制完整实现。
- 参考仓库提交：`37baab71b9abea1b76ab957409a1cc2fbfba8a26`。
- 后续 Coding Agent 必须先阅读 [AGENTS.md](AGENTS.md)。

用户已授权本次初始化和后续每个阶段在完成验证、更新学习记录后提交并推送到个人仓库。参考仓库保持独立且不纳入提交；每阶段同步完成后仍需等待学习者安排下一模块。

## 当前环境

| 项目 | 配置 |
| --- | --- |
| 系统 | Windows 11，64 位 |
| Python | 3.12.7，来自 `D:\Anaconda3\python.exe` |
| 项目入口 | `.venv\Scripts\python.exe` |
| PyTorch | 2.8.0+cpu |
| pytest | 7.4.4 |
| 运行设备 | `cpu` |

已创建项目 `.venv`，使用 `--system-site-packages` **只读复用**已有 Anaconda 的兼容 PyTorch 和 pytest，避免重新下载。解释器入口与本项目后续安装位置独立，但基础依赖仍来自 `D:\Anaconda3\Lib\site-packages`，并非完全隔离；移动项目到其他电脑时需重建环境，不要复制 `.venv`。不升级或卸载共享基础环境中的包。

PowerShell 下无需激活环境，直接运行项目解释器可避免执行策略问题：

```powershell
cd C:\Users\86185\Desktop\agent\mini-gpt-lab
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m scripts.check_environment
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

关闭 pytest 外部插件自动发现是为了避免继承全局环境中与本项目无关的插件；本项目测试只依赖 pytest 本身。`check_environment` 报告版本、解释器位置和设备，并打印实际 Shape：`[2, 3] @ [3, 4] → [2, 4]`，同时验证 CPU 反向传播。初始化验证结果：CPU 前向/反向均通过，pytest **1 passed**。

`PYTHONUTF8=1` 避免本次 Windows 终端捕获中文时出现乱码。当前测试结果为 **74 passed**，覆盖环境、Tokenizer、Embedding、Attention、FFN、Decoder Block、GPT 前向、标签、损失、生成和单步训练；尚未验证待填写的多轮循环。

只观察双头注意力的数据流，可运行 `.\.venv\Scripts\python.exe -m scripts.trace_forward --multi-head-only`。脚本中的固定小表仅用于演示，并与实际模块输出核对。

只观察 FFN，可运行 `.\.venv\Scripts\python.exe -m scripts.trace_forward --ffn-only`，查看升维、GELU、降维及逐位置独立性验证。

只查看 Decoder Block 的构造结果，可运行 `.\.venv\Scripts\python.exe -m scripts.trace_forward --block-only`。查看已实现 `forward` 的真实中间值，使用 `--block-forward-only`；此模式记录实际子模块调用，核对两次残差，不另行模拟未实现的计算。

查看字符编号到 GPT 词表分数的完整前向，可运行 `.\.venv\Scripts\python.exe -m scripts.trace_forward --gpt-only`。固定随机种子 42，记录实际 Embedding、各 Block、最终 LayerNorm 和 LM Head 输出；使用默认初始化的未训练参数，不执行生成。

查看移位标签与损失，可运行 `.\.venv\Scripts\python.exe -m scripts.trace_forward --loss-only`，显示完整编号、输入/标签、原始 logits、展平 Shape、正确字符概率、逐位置 loss 和平均 loss。概率仅用于展示，损失函数直接接收原始 logits；模型参数不更新。

查看未训练模型的字符生成，可运行 `.\.venv\Scripts\python.exe -m scripts.generate --prompt "cab a" --max-new-tokens 6 --trace`。此命令记录每轮真正送入模型的上下文、下一字符概率与选中编号；固定随机种子 42，输出 `"cab abcabcb"`，不更新参数。

单步参数更新可运行 `.\.venv\Scripts\python.exe -m scripts.train_tiny --steps 1`，同批次 loss 从 `1.335166` 降到 `1.143924`。`--steps 100` 是下一小节的练习入口，需先填好 `train_loop`；其四个固定窗口组成 `[4,8]` 输入与标签，并预留同一模型训练前后的生成对比。当前循环显式抛出 `NotImplementedError`，不作为训练完成结果。

本次 Codex 沙箱内的 Anaconda Python 曾以 DLL 初始化错误 `0xC0000142` 退出，Git 克隆也曾遇到 TLS 凭据错误；已通过获准的沙箱外执行完成检查与重试。这属于本次执行环境限制；普通本地终端可按上述命令验证。

## 安装与重建

依赖在 `pyproject.toml` 声明：运行时只需 PyTorch，开发依赖只有 pytest。已存在兼容基础环境时，可复用：

```powershell
& D:\Anaconda3\python.exe -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --no-index --no-build-isolation -e '.[dev]'
```

第二条命令使用本地已有依赖；基础环境需包含兼容的 torch、pytest 与 setuptools。若没有可复用环境，使用 Python 3.10 或以上创建完全独立的 `.venv`，安装 CPU 版 PyTorch：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
git clone https://github.com/karpathy/minGPT.git reference/minGPT
```

无需安装参考仓库的依赖，也无需下载任何预训练模型。`reference/`、`.venv/`、缓存、数据、权重、日志和输出不进入本项目版本控制。

## 目录与职责

```text
mini-gpt-lab/
├── reference/minGPT/          # 官方只读参考，包含自身 .git
├── mini_gpt/
│   ├── __init__.py
│   ├── tokenizer.py          # 已实现字符级编码与解码
│   ├── embeddings.py         # 已实现字符嵌入与学习式位置嵌入
│   ├── attention.py          # 已实现多头因果自注意力
│   ├── blocks.py             # 已实现 FFN 和 Pre-LN Decoder Block
│   ├── model.py              # 已实现 GPT 前向、LM Head 与无缓存贪心生成
│   └── losses.py             # 已实现移位标签与平均交叉熵
├── scripts/
│   ├── __init__.py
│   ├── check_environment.py  # 第 0 阶段环境验证
│   ├── trace_forward.py      # 已实现模块的真实数值与 Shape 追踪
│   ├── train_tiny.py         # 已实现单步训练，多轮循环为练习骨架
│   ├── generate.py           # 未训练模型生成入口与逐轮追踪
│   └── compare_cache.py      # 当前占位
├── tests/
│   ├── test_environment.py   # CPU 前向/反向 smoke test
│   ├── test_tokenizer.py     # 字符级 Tokenizer 测试
│   ├── test_embeddings.py    # 嵌入层数值、边界与梯度测试
│   ├── test_attention.py     # 单头及多头数值、因果性、批量与梯度测试
│   ├── test_ffn.py           # 非线性数值、逐位置独立性、参数共享与梯度测试
│   ├── test_decoder_block.py # Pre-LN 数值、残差、因果性、批量与梯度测试
│   ├── test_model.py         # 词表分数、输出头数值、堆叠因果性、长度与梯度测试
│   ├── test_losses.py        # 移位、独立损失数值、非连续张量、极端分数与梯度测试
│   ├── test_generate.py      # 生成回填、窗口裁剪、批量、边界与真实生成测试
│   └── test_training.py      # 单步更新、返回旧 loss 与清理历史梯度测试
├── data/
│   └── .gitkeep
├── docs/
│   ├── learning_log.md
│   ├── learning_plan.md
│   └── 01_tokenizer.md
├── AGENTS.md
├── README.md
├── pyproject.toml
└── .gitignore
```

另有本地 `.git/`、`.venv/` 和运行产生的忽略项。

## 两日学习顺序（按理解进度调整）

| 顺序 | 模块 | 最小理解与验证目标 |
| --- | --- | --- |
| 0 | 初始化 | 独立 Git、参考仓库、环境与 CPU smoke test |
| 1 | 字符级 Tokenizer | 词表、字符与 ID、encode/decode、未知字符与往返测试 |
| 2 | Embedding | `[B,T] → [B,T,C]`，Token 与学习式位置 Embedding 相加 |
| 3 | 手写 Attention | Q/K/V、分头、缩放点积、因果遮罩、合并多头 |
| 4 | FFN | `C → 4C → C` 与 GELU |
| 5 | Decoder Block | Pre-LN、两条残差、组合 Attention 与 FFN |
| 6 | 最小 GPT | LM Head、`[B,T,V]` logits、移位标签与损失 |
| 7 | 微型训练与生成 | CPU 短训练、逐字符自回归生成、上下文长度处理 |
| 8 | KV Cache | 缓存 K/V、位置偏移、与无缓存输出一致性 |

建议第 1 天学习 Tokenizer 至 Decoder Block，第 2 天完成模型组合、短训练、生成和最后的 KV Cache；时间分配服从实际理解程度。每一行单独讲解、实现、打印关键 Shape、测试并记录，不一次性写完。耗时较长训练须先得到确认。

当前完成课程 8.1 无缓存生成与 8.2.1 单步训练；8.2.2 多轮训练循环由学习者继续填写，完成后再验证训练前后的生成变化。KV Cache 留到最后。学习记录见 [docs/learning_log.md](docs/learning_log.md)。
