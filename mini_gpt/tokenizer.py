"""第 1 阶段练习：按小步骤实现字符级 Tokenizer。"""

from collections.abc import Iterable


class CharTokenizer:
    """使用固定字符词表，在字符与整数编号之间进行转换。"""

    def __init__(self, vocab: str) -> None:
        """输入词表字符串，例如 'abc '；按字符出现顺序分配编号。
           编号由词表中的位置决定，与字母表顺序无关
        例如：
            vocab = 'abc '
            编号： 0 1 2 3
            字符： a b c <space>
        """
        # 检查词表是否为空。
        if not vocab:
            raise ValueError("词表不能为空")
        # 检查词表是否包含重复字符。
        if len(set(vocab)) != len(vocab):
            raise ValueError("词表不能包含重复字符")
        # 建立 self.stoi，保存“字符 -> 编号”。
        self.stoi = {}
        for token_id, char in enumerate(vocab): #enumerate(vocab) 遍历能同时提供编号和字符
            self.stoi[char] = token_id #存对应关系到字典

        # 建立 self.itos，保存“编号 -> 字符”。
        self.itos = {}
        for token_id, char in enumerate(vocab):
            self.itos[token_id] = char #存对应关系到字典

    @property
    def vocab_size(self) -> int:
        """词表大小 V；合法编号从 0 到 V-1。"""
        # 返回词表中字符的数量。
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """输入长度为 T 的字符串，输出包含 T 个整数的 Python 列表。"""
        # 查表，把每个字符转换为编号。
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        """输入整数编号序列，输出对应的字符串。"""
        # 反向查表，并把字符拼接为字符串。
        return ''.join(self.itos[token_id] for token_id in token_ids)
