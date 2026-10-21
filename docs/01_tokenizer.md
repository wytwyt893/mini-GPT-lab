# 第 1 课：字符级 Tokenizer，先建立两张表

## 它在整条链路中的作用

模型接收数字，输入文本是字符序列。Tokenizer 负责建立字符与整数编号之间的对应关系。编号仅代表身份，不代表字符的大小、语义或相似度；把编号变成可训练向量是下一阶段 Embedding 的工作。

本轮采用固定词表，每个 Python 字符分配一个连续编号。词表写为 `"abc "` 时，最后一个字符是空格：

| 字符 | 编号 |
| --- | --- |
| `a` | 0 |
| `b` | 1 |
| `c` | 2 |
| 空格 | 3 |

`stoi` 表示字符到整数的映射，`itos` 表示整数到字符的映射。

若词表为 `vocab`，则第 `i` 个字符满足 `stoi[vocab[i]] = i`，反向满足 `itos[i] = vocab[i]`。这里是确定性的查表，不需要训练。

## 当前只写 __init__

打开 `mini_gpt/tokenizer.py`，只处理 TODO 1：

1. 使用参数 `vocab` 建立 `self.stoi`。
2. 建立反向表 `self.itos`。
3. 移除这个方法末尾的 `raise NotImplementedError(...)`。

提示：先在脑中展开 `enumerate("abc ")`，它会依次给出编号和字符；可以使用普通循环，也可以使用字典推导式。目前先假设词表非空且字符无重复，不补写其他方法。

验收样例位于 `tests/test_tokenizer.py`。完成后移除该测试上的 `@pytest.mark.skip(...)`，再执行：

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
.\.venv\Scripts\python.exe -m pytest tests/test_tokenizer.py -q
```

目前跳过标记表示练习尚未实现，不能视为功能通过。若尚未移除方法中的 `raise` 就启用测试，会得到有意保留的 `NotImplementedError`，含义是该练习仍待填写。

## 暂不实现的后续小步

依次学习 `vocab_size`、`encode`、`decode`，随后补充未知字符、重复词表、非法编号等边界检查。固定词表不能在生成时临时重排或添加字符，否则原有编号的含义会改变。

届时输入长度为 `T` 的文本，`encode` 输出长度为 `T` 的 Python 整数列表；列表没有张量 Shape，转成张量后才有 `[T]`，加批量维度后为 `[B,T]`。这些是预期约定，当前练习骨架尚未打印真实张量。

参考阅读：官方 [chargpt.py](../reference/minGPT/projects/chargpt/chargpt.py) 的 `CharDataset` 将字符映射嵌在数据集中；本项目将 Tokenizer 单独拆出练习，不复制完整数据集或训练实现。
