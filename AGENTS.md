# mini-gpt-lab：Coding Agent 教学约定

## 工作边界

- 仅在独立项目 `C:\Users\86185\Desktop\agent\mini-gpt-lab` 内操作，开始前核对目录和 Git 根目录。
- `reference/minGPT/` 是官方参考仓库，只读；不得编辑、格式化、训练或安装为可编辑依赖，不把其完整源码复制到本项目。
- 不修改其他项目或已有共享 Python 环境；依赖变更仅限本项目虚拟环境。
- 未经用户确认，不删除已有文件、不开展耗时较长的训练。
- 用户已于 2026-10-21 授权：本次初始化及后续每个学习阶段完成后，验证修改、更新学习记录、创建提交并推送到个人仓库 `https://github.com/wytwyt893/mini-GPT-lab.git` 的 `main` 分支，无需重复征询推送许可。
- 本项目根目录的 `origin` 仅绑定上述个人仓库；推送前核对 Git 根目录、远程地址及提交内容，不向 `karpathy/minGPT` 推送，不修改参考仓库的远程或源码。
- 不提交 `reference/`、`.venv/`、凭据、权重或运行生成文件；不强制推送、不覆盖远程历史。遇到远程新提交时先检查并保留已有内容，再同步。
- 当前只授权第 0 阶段初始化。完成后停止，等待用户安排第 1 阶段。

## 教学节奏

- 一次只实现一个学习模块。先说明计划与当前模块的目标，先让用户理解并明确同意继续，再进入下一模块。
- 每个模块解释功能、输入输出、必要公式与张量 Shape，并提供手算或微型示例。
- 统一形状符号：`B` 批量、`T` 序列长度、`C` 隐藏维度、`H` 头数、`D=C/H` 每头维度、`V` 词表大小。
- 关键位置在教学追踪脚本中打印真实张量维度；区分概念示意与真实运行结果。
- 每个实现的模块配套最小且有意义的单元测试；验证后说明通过项目与尚未覆盖的边界。
- 每阶段完成后更新 `docs/learning_log.md`，记录学习目标、公式/Shape、实现范围、验证结果和待确认问题。
- 遇到错误，先向用户解释原因，再修复；不默默跳过失败或虚报验证成功。

## 技术约定

- 使用 Python + PyTorch，Windows 本地 CPU 优先；显式选择 `cpu`，无需 CUDA、GPU 或大型模型权重。
- 独立实现更易理解的教学版 Decoder-only GPT，minGPT 只作为阅读参考。
- 手写 Multi-Head Causal Self-Attention，不用 `torch.nn.MultiheadAttention` 代替。
- 初期不用 FlashAttention、`scaled_dot_product_attention`、`torch.compile`、混合精度或复杂分布式训练。
- 经典 GPT：Pre-LN、GELU、学习式位置 Embedding、因果遮罩、残差连接。
- 先打通完整前向和不带缓存的自回归生成，最后单独学习 KV Cache；占位文件不等于已经实现。
- 不安装 Transformers、Lightning 等无必要的大框架，不自动下载数据集或预训练权重。
- 使用项目 `.venv\Scripts\python.exe` 运行脚本和 `python -m pytest`；若复用共享依赖，只读共享包，不升级或卸载共享环境。

## 建议模块顺序

0. 工程初始化与环境验证（本轮）。
1. 自定义字符级 Tokenizer：词表、encode/decode、未知字符策略和往返测试。
2. Token Embedding 与学习式位置 Embedding。
3. Multi-Head Causal Self-Attention。
4. FFN（GELU）。
5. Pre-LN Decoder Block。
6. 最小 GPT 与 LM Head、移位标签和 next-token loss。
7. 小规模 CPU 训练与无缓存自回归生成。
8. KV Cache 及与无缓存输出的一致性验证。
