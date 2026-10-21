"""练习 1 的验收样例；当前跳过不代表 Tokenizer 已实现或测试通过。"""

import pytest

from mini_gpt.tokenizer import CharTokenizer


# 完成 __init__ 后移除下面的 skip 标记，再运行本测试进行验收。
@pytest.mark.skip(reason="练习 1 待实现；完成 __init__ 后取消跳过")
def test_character_vocabulary_mapping():
    tokenizer = CharTokenizer("abc ")
    assert tokenizer.stoi == {"a": 0, "b": 1, "c": 2, " ": 3}
    assert tokenizer.itos == {0: "a", 1: "b", 2: "c", 3: " "}
