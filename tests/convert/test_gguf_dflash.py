"""dflash draft GGUF: the derived config, the tensor table, the refusals, and the CLI wiring.

gguf-py is a test-only dependency here (the converter reads GGUF natively), so the module skips
when it is not importable. The tiny draft is written by gguf-py's own writer, which is what makes
the shape checks meaningful: the numbers below come from the container, not from the reader.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gguf")
import gguf  # noqa: E402

from tools.convert import gguf_dflash  # noqa: E402
from tests.convert.test_gguf_qwen4exp import _write_tiny  # noqa: E402

HIDDEN, VOCAB, TARGET_LAYERS = 64, 32, 2
DRAFT_LAYERS, HEADS, KV, HEAD_DIM = 1, 4, 2, 8
INTER, KERNEL, GROUP, RANK, TOP_K = 96, 2, 4, 4, 2
TAPS = [2]  # llama.cpp's 1-based boundaries: the target's zero-based block 1
MASK = 7

TEXT = {"hidden_size": HIDDEN, "vocab_size": VOCAB, "num_hidden_layers": TARGET_LAYERS}


def _t(rng, shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float16)


def _write_draft(dir_path, *, extra=None, drop=()):
    rng = np.random.default_rng(5)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "draft.gguf"
    w = gguf.GGUFWriter(str(path), "dflash")
    w.add_block_count(DRAFT_LAYERS)
    w.add_context_length(4096)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV)
    w.add_causal_attention(False)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_block_size(8)
    w.add_conv_kernel_size(KERNEL)
    w.add_conv_group_size(GROUP)
    w.add_selector_rank(RANK)
    w.add_selector_top_k(TOP_K)
    w.add_target_layers(TAPS)
    w.add_sliding_window(1024)
    w.add_sliding_window_pattern([True] * DRAFT_LAYERS)
    # The degenerate M-RoPE sections llama.cpp writes for a draft whose target ropes in 3 axes.
    w.add_rope_dimension_sections([HEAD_DIM // 2, 0, 0, 0])
    w.add_mask_token_id(MASK)

    q, k = HEADS * HEAD_DIM, KV * HEAD_DIM
    projected = 2 * KERNEL * (HIDDEN // GROUP)
    tensors = {
        "fc.weight": _t(rng, (HIDDEN, len(TAPS) * HIDDEN)),
        "enc.output_norm.weight": _t(rng, (HIDDEN,)),
        "output_norm.weight": _t(rng, (HIDDEN,)),
        "selector_hidden.weight": _t(rng, (RANK, HIDDEN)),
        "selector_predecessor.weight": _t(rng, (VOCAB, RANK)),
        "selector_successor.weight": _t(rng, (VOCAB, RANK)),
    }
    for layer in range(DRAFT_LAYERS):
        b = f"blk.{layer}."
        tensors.update(
            {
                b + "attn_norm.weight": _t(rng, (HIDDEN,)),
                b + "ffn_norm.weight": _t(rng, (HIDDEN,)),
                b + "attn_q.weight": _t(rng, (q, HIDDEN)),
                b + "attn_k.weight": _t(rng, (k, HIDDEN)),
                b + "attn_v.weight": _t(rng, (k, HIDDEN)),
                b + "attn_output.weight": _t(rng, (HIDDEN, q)),
                b + "attn_q_norm.weight": _t(rng, (HEAD_DIM,)),
                b + "attn_k_norm.weight": _t(rng, (HEAD_DIM,)),
                b + "ffn_gate.weight": _t(rng, (INTER, HIDDEN)),
                b + "ffn_up.weight": _t(rng, (INTER, HIDDEN)),
                b + "ffn_down.weight": _t(rng, (HIDDEN, INTER)),
                b + "attn_conv_base": _t(rng, (2, KERNEL, HIDDEN)),
                b + "attn_conv_proj.weight": _t(rng, (projected, HIDDEN)),
                b + "ffn_conv_base": _t(rng, (2, KERNEL, HIDDEN)),
                b + "ffn_conv_proj.weight": _t(rng, (projected, HIDDEN)),
            }
        )
    tensors.update(extra or {})
    for name, data in tensors.items():
        if name in drop:
            continue
        w.add_tensor(name, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def test_config_is_derived_from_the_file_and_the_target(tmp_path):
    path = _write_draft(tmp_path)
    with gguf_dflash.open_companion(path, TEXT, backend="dflash2") as source:
        config = source.config
        assert config["architectures"] == ["DFlash2DraftModel"]
        assert config["model_type"] == "qwen3"
        assert config["num_hidden_layers"] == DRAFT_LAYERS
        assert config["num_attention_heads"] == HEADS
        assert config["num_key_value_heads"] == KV
        assert config["head_dim"] == HEAD_DIM
        assert config["intermediate_size"] == INTER
        assert config["max_position_embeddings"] == 4096
        assert config["rope_parameters"] == {"rope_theta": 10000.0}
        assert config["layer_types"] == ["sliding_attention"]
        assert config["sliding_window"] == 1024
        draft = config["dflash_config"]
        # llama.cpp stores 1-based boundaries; NInfer conditions on zero-based block ids, so the
        # value must come back shifted or every tap reads one block late.
        assert draft["target_layer_ids"] == [index - 1 for index in TAPS]
        assert draft["mask_token_id"] == MASK
        assert (draft["conv_kernel_size"], draft["conv_group_size"]) == (KERNEL, GROUP)
        assert (draft["selector_rank"], draft["selector_top_k"]) == (RANK, TOP_K)
        # Every tensor of the file is claimed by the table: nothing is dropped on the floor.
        assert set(source.pairs.values()) == set(source.file.tensors)
        assert source.describe("fc.weight").shape == (HIDDEN, len(TAPS) * HIDDEN)
        assert source.describe("layers.0.attention_conv.base_kernel").shape == (
            2,
            KERNEL,
            HIDDEN,
        )
        assert source.describe("candidate_selector.predecessor_codebook").shape == (
            VOCAB,
            RANK,
        )


def test_an_unmapped_tensor_is_refused_by_name(tmp_path):
    path = _write_draft(
        tmp_path,
        extra={"blk.0.unexpected_projection.weight": _t(np.random.default_rng(1), (HIDDEN,))},
    )
    with pytest.raises(ValueError, match="does not map"):
        gguf_dflash.open_companion(path, TEXT, backend="dflash2").close()


def test_a_missing_tensor_is_refused_by_name(tmp_path):
    path = _write_draft(tmp_path, drop={"selector_successor.weight"})
    with pytest.raises(ValueError, match="selector_successor.weight"):
        gguf_dflash.open_companion(path, TEXT, backend="dflash2").close()


def test_a_dflash2_file_cannot_be_named_dflash(tmp_path):
    path = _write_draft(tmp_path)
    with pytest.raises(ValueError, match="DFlash2 metadata"):
        gguf_dflash.open_companion(path, TEXT, backend="dflash").close()


def test_a_draft_of_another_width_is_refused(tmp_path):
    path = _write_draft(tmp_path)
    with pytest.raises(ValueError, match="the draft is .* wide"):
        gguf_dflash.open_companion(
            path,
            {**TEXT, "hidden_size": HIDDEN * 2},
            backend="dflash2",
        ).close()


def test_cli_dry_run_wires_the_draft_into_the_qwen4exp_artifact(tmp_path, capsys):
    from tools.convert.__main__ import main as convert_main

    base, _ = _write_tiny(tmp_path)
    draft = _write_draft(tmp_path / "draft")
    out = tmp_path / "plan.ninfer"
    convert_main(
        [
            "--model", str(base),
            "--recipe", "qwen4_exp",
            "--components", "text,dflash2",
            "--dflash2-gguf", str(draft),
            "--dry-run",
            "--device", "cpu",
            "--out", str(out),
        ]
    )
    printed = capsys.readouterr().out
    assert "[dry-run] dflash2: " in printed
    assert "candidate_selector.successor_codebook <- selector_successor.weight" in printed
    assert "nothing written" in printed
    # A dry run must not leave an artifact or its report behind.
    assert not out.exists()
    assert not out.with_name(out.name + ".conversion.json").exists()
