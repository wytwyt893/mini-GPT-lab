"""保证入门导览展示真实生成数据，且观察器不会影响模型后续调用。"""

import sys

import pytest
import torch

from mini_gpt.tokenizer import CharTokenizer
from scripts.baby_flow import capture_flow, main, make_demo_model, print_flow


@pytest.mark.parametrize("use_cache", [False, True])
def test_observing_generation_preserves_known_predictions(use_cache):
    model = make_demo_model()
    # 将分数全部设为 0，使预期编号独立于随机初始化。
    with torch.no_grad():
        model.lm_head.weight.zero_()
    run = capture_flow(model, CharTokenizer("abc "), "bc", 2, use_cache)
    assert run.tokenizer.decode(run.output_ids[0].tolist()) == "bcaa"
    for probabilities in run.trace.generation.values["probs"]:
        torch.testing.assert_close(probabilities, torch.full((1, 4), 0.25))
    generate = model.generate_cached if use_cache else model.generate
    torch.testing.assert_close(generate(run.input_ids, 2), run.output_ids)


def test_attention_record_keeps_intermediates_before_they_are_overwritten():
    run = capture_flow(make_demo_model(), CharTokenizer("abc "), "cab", 1)
    attention = run.trace.rounds[0].calls["blocks.0.attn"]
    raw, scaled, masked = attention.values["scores"]
    assert raw.shape == scaled.shape == masked.shape == (1, 2, 3, 3)
    assert torch.isfinite(raw).all() and torch.isfinite(scaled).all()
    assert torch.isneginf(masked[0, :, 0, 1:]).all()
    assert torch.count_nonzero(attention.last("weights")[0, :, 0, 1:]) == 0
    per_head, merged = attention.values["context"]
    assert per_head.shape == (1, 2, 3, 2)
    assert merged.shape == (1, 3, 4)


def test_cache_trace_shows_new_position_query_and_all_historical_keys():
    run = capture_flow(make_demo_model(), CharTokenizer("abc "), "cab", 2, use_cache=True)
    second = run.trace.rounds[1]
    assert second.input_ids.shape == (1, 1)
    assert second.calls["embedding"].last("position_ids").tolist() == [3]
    attention = second.calls["blocks.0.attn"]
    assert attention.last("q").shape == (1, 2, 1, 2)
    assert attention.last("k").shape == attention.last("v").shape == (1, 2, 4, 2)
    assert attention.last("mask").tolist() == [[True, True, True, True]]


def test_sliding_window_trace_shows_a_real_cache_rebuild():
    run = capture_flow(make_demo_model(), CharTokenizer("abc "), "abc abc ", 2, use_cache=True)
    for record in run.trace.rounds:
        assert record.input_ids.shape == (1, 8)
        assert record.calls[""].last("past_length") == 0
        assert record.calls["embedding"].last("position_ids").tolist() == list(range(8))


@pytest.mark.parametrize("all_rounds", [False, True])
def test_terminal_guide_names_owners_and_decodes_actual_result(capsys, all_rounds):
    run = capture_flow(make_demo_model(), CharTokenizer("abc "), "cab", 2, use_cache=True)
    print_flow(run, all_rounds=all_rounds)
    output = capsys.readouterr().out
    for filename in ("tokenizer.py", "embeddings.py", "attention.py", "blocks.py", "model.py", "cache.py"):
        assert filename in output
    for concept in ("白话解释", "因果遮罩", "LN1", "FFN", "LM Head", "实际 Softmax", "自检通过"):
        assert concept in output
    assert f"最终完整文本：{run.tokenizer.decode(run.output_ids[0].tolist())!r}" in output
    assert output.count("送给 Attention 的 LN1 输出") == (2 if all_rounds else 1)
    assert ("本轮仍实际经过" in output) is (not all_rounds)


def test_observer_restores_tracing_and_hooks_even_if_generation_fails():
    model = make_demo_model()
    previous_trace = sys.gettrace()
    previous_counts = [len(module._forward_hooks) for module in model.modules()]

    def stop_generation(module, args):
        raise RuntimeError("演示异常退出")

    stop_handle = model.lm_head.register_forward_pre_hook(stop_generation)
    try:
        with pytest.raises(RuntimeError, match="演示异常退出"):
            capture_flow(model, CharTokenizer("abc "), "cab", 1)
    finally:
        stop_handle.remove()
    assert sys.gettrace() is previous_trace
    assert [len(module._forward_hooks) for module in model.modules()] == previous_counts


@pytest.mark.parametrize("arguments", [
    ["--prompt", ""],
    ["--prompt", "z"],
    ["--max-new-tokens", "0"],
])
def test_cli_reports_inputs_that_cannot_be_demonstrated(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert capsys.readouterr().err
