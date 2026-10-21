# 学习记录

## 2026-10-21：第 0 阶段——工程初始化

### 本阶段目标与范围

建立独立的 `mini-gpt-lab`、克隆只读官方参考、创建占位目录和最小 CPU 学习环境。没有实现任何 GPT 学习模块，没有训练、下载权重或实现 KV Cache。

### 工程与环境

- 路径：`C:\Users\86185\Desktop\agent\mini-gpt-lab`；初始目录为空且不属于其他 Git 仓库。
- Git 2.51.0.windows.1；已执行 `git init`，未创建远程、提交或推送。
- 官方参考 `https://github.com/karpathy/minGPT.git` 已克隆到 `reference/minGPT/`。
- 参考提交 `37baab71b9abea1b76ab957409a1cc2fbfba8a26`；仅阅读，源码未修改。
- Windows 11、64 位；Python 3.12.7；PyTorch 2.8.0+cpu；pytest 7.4.4；设备 `cpu`。
- `.venv` 通过 `D:\Anaconda3\python.exe -m venv --system-site-packages .venv` 创建，只读复用兼容依赖，不修改共享环境，不重复下载 CPU PyTorch。
- 声明最小依赖 torch 与 pytest；README 记录复用方式与完全独立环境的重建方式。
- `.gitignore` 排除参考仓库、虚拟环境、缓存、模型权重、日志、数据和运行输出。
- `AGENTS.md` 固化逐模块教学、Shape/公式、最小测试、每阶段记录和用户确认规则。

### 验证与实际 Shape

环境验证仅使用普通矩阵乘法和自动求导，并非实现 Attention 或其他 GPT 模块：

```text
x:          [2, 3]  float32, CPU
weight:     [3, 4]  float32, CPU，全部为 1
y = x @ W:  [2, 4]
loss:       标量，y.sum()
x.grad:     [2, 3]，每个元素为 4
```

`x=[[0,1,2],[3,4,5]]`，输出两行分别全为 3 和 12；每个输入元素参与 4 个输出求和，因此对 `loss` 的梯度为 4。以这一小例子验证 CPU 前向和反向传播。

实际执行：项目 `.venv\Scripts\python.exe scripts/check_environment.py` 通过，报告 `is_virtual_environment=true`、设备 `cpu`、CPU 前向/反向均通过；关闭外部 pytest 插件自动发现后，`python -m pytest -q` 返回 **1 passed in 1.61s**。README 使用等价的模块运行入口。

本地离线可编辑安装 `python -m pip install --no-index --no-build-isolation -e '.[dev]'` 成功，安装 `mini-gpt-lab==0.0.0` 到本项目环境；所有第三方依赖复用已有版本，无额外下载。`git check-ignore` 验证参考仓库、虚拟环境、缓存、权重、日志及输出均被忽略；参考仓库 `git status --porcelain` 为空，本项目 Git 根目录正确且没有远程。

### 遇到的问题与处理

- 沙箱内 Git 克隆发生 Windows TLS `SEC_E_NO_CREDENTIALS`；解释原因后，获准在沙箱外重试并成功。
- 沙箱内 Anaconda Python 启动返回 `-1073741790`（`0xC0000142`，DLL 初始化错误）；通过获准的沙箱外执行验证环境，未改动系统设置。
- 首次内联 Python 探测受到 Windows 命令行引号传递影响，触发 `NameError`；改用项目内独立脚本进行可复现检查。
- 不使用需要额外系统权限的 WMI 查询，改用 Python 标准库确认操作系统。
- 首次文档补丁漏写新增行标记，校验拒绝写入；补齐后重新创建文档。

### 下一阶段（未开始）

建议第 1 阶段：自定义字符级 Tokenizer。先解释词表、`stoi/itos`、`encode/decode`、未知字符策略，以及字符串长度与 `[T]` token ID 序列的对应关系，再实现最小往返测试。等待学习者安排后才开始。

## 2026-10-21：个人 GitHub 仓库同步

- 用户提供个人仓库 `https://github.com/wytwyt893/mini-GPT-lab.git`，并明确授权本次初始化与后续各阶段完成后提交、推送。
- 经远程查询，首次同步前该仓库为空；本地学习项目独立于 `reference/minGPT/`，项目 `origin` 绑定个人仓库，使用 `main` 分支。
- 首次提交范围为第 0 阶段项目骨架、依赖配置、环境验证、README、教学规则及学习记录，不包含参考仓库、虚拟环境、缓存或运行输出。
- 后续固定流程：完成当前模块 → 验证 → 更新学习记录 → 检查提交内容及远程 → commit → push 到个人仓库；无需重复请求推送许可。
- 参考仓库的官方远程和源码保持原样；同步操作不推进 Tokenizer 或其他学习模块。
