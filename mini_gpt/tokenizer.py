"""第 1 阶段练习：按小步骤实现字符级 Tokenizer。"""

from collections.abc import Iterable


class CharTokenizer:
    """字符与整数编号之间的转换器；目前只保留接口和中文 TODO。"""

    def __init__(self, vocab: str) -> None:
        """输入词表字符串，例如 'abc '；按字符出现顺序分配编号。"""
        # TODO 1：建立 self.stoi，保存“字符 -> 编号”。
        # TODO 1：建立 self.itos，保存“编号 -> 字符”。
        # 提示：enumerate(vocab) 能同时提供编号和字符。
        # 先只处理非空、无重复字符的词表，边界检查稍后补充。
        raise NotImplementedError("练习 1：请先完成两张映射表")

    @property
    def vocab_size(self) -> int:
        """词表大小 V；合法编号从 0 到 V-1。"""
        # TODO 2：返回词表中字符的数量。
        raise NotImplementedError("练习 2：稍后实现词表大小")

    def encode(self, text: str) -> list[int]:
        """输入长度为 T 的字符串，输出包含 T 个整数的 Python 列表。"""
        # TODO 3：查表，把每个字符转换为编号。
        raise NotImplementedError("练习 3：稍后实现编码")

    def decode(self, token_ids: Iterable[int]) -> str:
        """输入整数编号序列，输出对应的字符串。"""
        # TODO 4：反向查表，并把字符拼接为字符串。
        raise NotImplementedError("练习 4：稍后实现解码")
