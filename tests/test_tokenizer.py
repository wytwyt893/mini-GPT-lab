"""验证学习者已实现的字符映射、词表大小、编码与解码。"""

import pytest

from mini_gpt.tokenizer import CharTokenizer


def test_character_vocabulary_mapping():
    tokenizer = CharTokenizer("abc ")
    assert tokenizer.stoi == {"a": 0, "b": 1, "c": 2, " ": 3}
    assert tokenizer.itos == {0: "a", 1: "b", 2: "c", 3: " "}
    assert tokenizer.vocab_size == 4


def test_ids_follow_vocabulary_order():
    # 验证编号取决于输入词表的位置，不依赖字母表顺序。
    tokenizer = CharTokenizer("cba ")
    assert tokenizer.stoi["a"] == 2
    assert tokenizer.itos[2] == "a"


def test_known_encoding_and_decoding():
    # 分别验证准确结果，避免只做往返测试而漏掉编号错误。
    tokenizer = CharTokenizer("abc ")
    assert tokenizer.encode("cab a") == [2, 0, 1, 3, 0]
    assert tokenizer.decode([2, 0, 1, 3, 0]) == "cab a"


@pytest.mark.parametrize("text", ["", "a", "a a", "你好\na"])
def test_round_trip_preserves_text(text):
    # 空串、重复字符、空格、中文和换行均应原样还原。
    tokenizer = CharTokenizer("ab 你好\n")
    token_ids = tokenizer.encode(text)
    assert isinstance(token_ids, list)
    assert len(token_ids) == len(text)
    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert all(0 <= token_id < tokenizer.vocab_size for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


@pytest.mark.parametrize("vocab", ["aa", "aba"])
def test_duplicate_vocabulary_is_rejected(vocab):
    with pytest.raises(ValueError, match="词表不能包含重复字符"):
        CharTokenizer(vocab)


def test_unknown_character_is_rejected():
    # 当前采用严格词表策略，未知字符由字典查找抛出 KeyError。
    with pytest.raises(KeyError):
        CharTokenizer("abc ").encode("a!")


@pytest.mark.parametrize("token_id", [-1, 4])
def test_unknown_token_id_is_rejected(token_id):
    with pytest.raises(KeyError):
        CharTokenizer("abc ").decode([token_id])


def test_empty_vocabulary_is_rejected():
    with pytest.raises(ValueError, match="词表不能为空"):
        CharTokenizer("")
