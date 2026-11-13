"""GGUF input for Qwen3.5 dense: the qwen35 mapping, the tone of its norms, and a real conversion.

The tiny file below is written with gguf-py's writer, so the reader is exercised against the
implementation that defines the format. The end-to-end arm then runs the real converter and
checks the artifact's contents against the values the mapping promises: a norm lands as `w - 1`,
the GDN value rows land in HF head order, and the convolution lands transposed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gguf")
import gguf  # noqa: E402
from gguf.constants import GGUFValueType  # noqa: E402

from tools.convert.gguf_qwen3_5 import GGUFSource, open_vision_source  # noqa: E402

HIDDEN, VOCAB, LAYERS = 64, 32, 2
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 8
NK, NV, DK, DV = 2, 4, 8, 8
TAPS, INTER, INTERVAL = 3, 32, 2
Q, K = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
V, C = NV * DV, 2 * NK * DK + NV * DV
ROPE = 2  # head_dim * partial_rotary_factor (0.25)
SECTIONS = [1, 0, 0]


def _t(rng, shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _write_tiny(dir_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """One linear layer, one full-attention layer, then the nextn draft block."""
    rng = np.random.default_rng(7)
    blocks = LAYERS + 1
    model = dir_path / "tiny.gguf"
    tensors: dict[str, np.ndarray] = {}
    w = gguf.GGUFWriter(str(model), "qwen35")

    w.add_block_count(blocks)
    w.add_context_length(128)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(INTER)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10_000_000.0)
    w.add_rope_dimension_count(ROPE)
    w.add_rope_dimension_sections(SECTIONS)
    w.add_full_attention_interval(INTERVAL)
    w.add_nextn_predict_layers(1)
    w.add_ssm_group_count(NK)
    w.add_ssm_time_step_rank(NV)
    w.add_ssm_state_size(DK)
    w.add_ssm_inner_size(NV * DV)
    w.add_ssm_conv_kernel(TAPS)

    def add(name, data):
        tensors[name] = data
        w.add_tensor(name, data)

    add("token_embd.weight", _t(rng, (VOCAB, HIDDEN)))
    add("output.weight", _t(rng, (VOCAB, HIDDEN)))
    add("output_norm.weight", 1 + _t(rng, (HIDDEN,)))

    for layer in range(LAYERS):
        prefix = f"blk.{layer}."
        add(prefix + "attn_norm.weight", 1 + _t(rng, (HIDDEN,)))
        add(prefix + "post_attention_norm.weight", 1 + _t(rng, (HIDDEN,)))
        if layer % INTERVAL == INTERVAL - 1:
            add(prefix + "attn_q.weight", _t(rng, (2 * Q, HIDDEN)))
            add(prefix + "attn_k.weight", _t(rng, (K, HIDDEN)))
            add(prefix + "attn_v.weight", _t(rng, (K, HIDDEN)))
            add(prefix + "attn_output.weight", _t(rng, (HIDDEN, Q)))
            add(prefix + "attn_q_norm.weight", 1 + _t(rng, (HEAD_DIM,)))
            add(prefix + "attn_k_norm.weight", 1 + _t(rng, (HEAD_DIM,)))
        else:
            add(prefix + "attn_qkv.weight", _t(rng, (C, HIDDEN)))
            add(prefix + "attn_gate.weight", _t(rng, (V, HIDDEN)))
            add(prefix + "ssm_out.weight", _t(rng, (HIDDEN, V)))
            add(prefix + "ssm_alpha.weight", _t(rng, (NV, HIDDEN)))
            add(prefix + "ssm_beta.weight", _t(rng, (NV, HIDDEN)))
            add(prefix + "ssm_conv1d.weight", _t(rng, (C, TAPS)))
            a_log = _t(rng, (NV,))
            tensors[prefix + "A_log"] = a_log
            add(prefix + "ssm_a", -np.exp(a_log))
            add(prefix + "ssm_dt.bias", _t(rng, (NV,)))
            add(prefix + "ssm_norm.weight", 1 + _t(rng, (DK,)))
        add(prefix + "ffn_gate.weight", _t(rng, (INTER, HIDDEN)))
        add(prefix + "ffn_up.weight", _t(rng, (INTER, HIDDEN)))
        add(prefix + "ffn_down.weight", _t(rng, (HIDDEN, INTER)))

    prefix = f"blk.{LAYERS}."
    add(prefix + "attn_norm.weight", 1 + _t(rng, (HIDDEN,)))
    add(prefix + "post_attention_norm.weight", 1 + _t(rng, (HIDDEN,)))
    add(prefix + "attn_q.weight", _t(rng, (2 * Q, HIDDEN)))
    add(prefix + "attn_k.weight", _t(rng, (K, HIDDEN)))
    add(prefix + "attn_v.weight", _t(rng, (K, HIDDEN)))
    add(prefix + "attn_output.weight", _t(rng, (HIDDEN, Q)))
    add(prefix + "attn_q_norm.weight", 1 + _t(rng, (HEAD_DIM,)))
    add(prefix + "attn_k_norm.weight", 1 + _t(rng, (HEAD_DIM,)))
    add(prefix + "ffn_gate.weight", _t(rng, (INTER, HIDDEN)))
    add(prefix + "ffn_up.weight", _t(rng, (INTER, HIDDEN)))
    add(prefix + "ffn_down.weight", _t(rng, (HIDDEN, INTER)))
    add(prefix + "nextn.eh_proj.weight", _t(rng, (HIDDEN, 2 * HIDDEN)))
    add(prefix + "nextn.enorm.weight", 1 + _t(rng, (HIDDEN,)))
    add(prefix + "nextn.hnorm.weight", 1 + _t(rng, (HIDDEN,)))
    add(prefix + "nextn.shared_head_norm.weight", 1 + _t(rng, (HIDDEN,)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return model, tensors


def _write_mmproj(dir_path: Path) -> Path:
    """A one-block clip projector: patch embed, one block, merger."""
    rng = np.random.default_rng(11)
    depth, hidden, inter, heads, patch, merge = 1, 8, 16, 2, 2, 2
    positions = 4
    model = dir_path / "mmproj.gguf"
    w = gguf.GGUFWriter(str(model), "clip")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "qwen3vl_merger")
    w.add_uint32("clip.vision.block_count", depth)
    w.add_uint32("clip.vision.embedding_length", hidden)
    w.add_uint32("clip.vision.feed_forward_length", inter)
    w.add_uint32("clip.vision.attention.head_count", heads)
    w.add_uint32("clip.vision.patch_size", patch)
    w.add_uint32("clip.vision.spatial_merge_size", merge)
    w.add_uint32("clip.vision.projection_dim", HIDDEN)

    def add(name, data):
        w.add_tensor(name, data)

    merged = merge * merge * hidden
    add("v.patch_embd.weight", _t(rng, (hidden, 3, patch, patch)))
    add("v.patch_embd.weight.1", _t(rng, (hidden, 3, patch, patch)))
    add("v.patch_embd.bias", _t(rng, (hidden,)))
    add("v.position_embd.weight", _t(rng, (positions, hidden)))
    add("v.post_ln.weight", 1 + _t(rng, (hidden,)))
    add("v.post_ln.bias", _t(rng, (hidden,)))
    for index in range(depth):
        prefix = f"v.blk.{index}."
        for role in ("ln1", "ln2"):
            add(prefix + role + ".weight", 1 + _t(rng, (hidden,)))
            add(prefix + role + ".bias", _t(rng, (hidden,)))
        add(prefix + "attn_qkv.weight", _t(rng, (3 * hidden, hidden)))
        add(prefix + "attn_qkv.bias", _t(rng, (3 * hidden,)))
        add(prefix + "attn_out.weight", _t(rng, (hidden, hidden)))
        add(prefix + "attn_out.bias", _t(rng, (hidden,)))
        add(prefix + "ffn_up.weight", _t(rng, (inter, hidden)))
        add(prefix + "ffn_up.bias", _t(rng, (inter,)))
        add(prefix + "ffn_down.weight", _t(rng, (hidden, inter)))
        add(prefix + "ffn_down.bias", _t(rng, (hidden,)))
    add("mm.0.weight", _t(rng, (merged, merged)))
    add("mm.0.bias", _t(rng, (merged,)))
    add("mm.2.weight", _t(rng, (HIDDEN, merged)))
    add("mm.2.bias", _t(rng, (HIDDEN,)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return model


def _resources(dir_path: Path) -> Path:
    """The frontend resources the builder needs (a minimalist but valid tokenizer)."""
    root = dir_path / "root"
    root.mkdir(exist_ok=True)
    vocabulary = {f"t{index}": index for index in range(VOCAB - 1)}
    vocabulary["<|endoftext|>"] = VOCAB - 1
    tokenizer = {"model": {"vocab": vocabulary}, "added_tokens": [
        {"id": VOCAB - 1, "content": "<|endoftext|>", "special": True}
    ]}
    (root / "tokenizer.json").write_text(json.dumps(tokenizer))
    (root / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": {}}))
    (root / "chat_template.jinja").write_text("{{ messages }}")
    (root / "generation_config.json").write_text("{}")
    return root


def _perm(nv: int, nk: int, per: int, dim: int) -> np.ndarray:
    return np.transpose(np.arange(nv * dim).reshape(per, nk, dim), (1, 0, 2)).reshape(-1)


def test_source_maps_qwen35_names(tmp_path):
    model, tensors = _write_tiny(tmp_path)
    root = _resources(tmp_path)
    with GGUFSource(model, root=root) as source:
        text = source.config["text_config"]
        assert text["num_hidden_layers"] == LAYERS
        assert text["layer_types"] == ["linear_attention", "full_attention"]
        assert text["hidden_size"] == HIDDEN
        assert text["vocab_size"] == VOCAB
        assert text["linear_num_key_heads"] == NK
        assert text["linear_value_head_dim"] == DV
        assert text["linear_conv_kernel_dim"] == TAPS
        assert text["rope_parameters"]["mrope_section"] == SECTIONS
        # Unit-offset norms: the artifact keeps w, llama.cpp wrote w + 1.
        got = source.read_flat("model.layers.0.input_layernorm.weight").float().numpy()
        assert np.allclose(got, tensors["blk.0.attn_norm.weight"] - 1, atol=1e-2)
        # GDN value rows in HF head order.
        got = source.read_flat(
            "model.layers.0.linear_attn.in_proj_qkv.weight"
        ).float().numpy().reshape(C, HIDDEN)
        qkv = tensors["blk.0.attn_qkv.weight"]
        assert np.allclose(got[: 2 * NK * DK], qkv[: 2 * NK * DK], atol=1e-2)
        assert np.allclose(got[2 * NK * DK:], qkv[2 * NK * DK:][_perm(NV, NK, NV // NK, DV)],
                           atol=1e-2)
        # A_log is log(-ssm_a); the convolution is transposed with the V channels re-ordered.
        got = source.read_flat("model.layers.0.linear_attn.A_log").float().numpy()
        assert np.allclose(got, tensors["blk.0.A_log"][_perm(NV, NK, NV // NK, 1)],
                           atol=2e-3)
        got = source.read_flat("model.layers.0.linear_attn.conv1d.weight").float().numpy()
        conv = tensors["blk.0.ssm_conv1d.weight"]
        expected = np.concatenate(
            [conv[: 2 * NK * DK],
             conv[2 * NK * DK:][_perm(NV, NK, NV // NK, DV)]]
        ).reshape(C, TAPS)
        assert np.allclose(got.reshape(C, TAPS), expected, atol=2e-3)
        assert source.code_format("model.layers.0.mlp.down_proj.weight") is None


def test_vision_source_maps_the_tower(tmp_path):
    model, _ = _write_tiny(tmp_path)
    root = _resources(tmp_path)
    mmproj = _write_mmproj(tmp_path)
    with GGUFSource(model, root=root) as source:
        hidden = int(source.config["text_config"]["hidden_size"])
    with open_vision_source(mmproj, hidden, root=root) as vision:
        cfg = vision.config["vision_config"]
        assert cfg["depth"] == 1
        assert cfg["temporal_patch_size"] == 2
        assert cfg["num_position_embeddings"] == 4
        shape = vision.describe("model.visual.patch_embed.proj.weight").shape
        assert shape == (8, 3, 2, 2, 2)


def test_conversion_from_gguf(tmp_path):
    torch = pytest.importorskip("torch")
    from tools.artifact.reader import Artifact
    from tools.convert.__main__ import main as convert_main

    model, tensors = _write_tiny(tmp_path)
    root = _resources(tmp_path)
    out = tmp_path / "tiny.ninfer"
    convert_main([
        "--model", str(model),
        "--recipe", "qwen3_8_27b_gguf",
        "--components", "text",
        "--gguf-root", str(root),
        "--device", "cpu",
        "--name", "tiny-qwen3_5",
        "--out", str(out),
    ])
    report = json.loads(Path(str(out) + ".conversion.json").read_text())
    by_param = {}
    for method in report["methods"]:
        for name in method["parameters"]:
            by_param[name] = method
    with Artifact.open(out) as artifact:
        # The role table decides the format of a fused parent, not the source alone: the attention
        # pair is Q4/Q5 because that is the only paired native form.
        assert by_param["text/layers/1/attention/query"]["format"] == "q4_g64_fp16"
        assert by_param["text/layers/1/attention/gate"]["format"] == "q5_g64_fp16"
        assert by_param["text/layers/0/gdn/query"]["format"] == "q4_g64_fp16"
        assert by_param["text/layers/0/gdn/value"]["format"] == "q5_g64_fp16"
        assert by_param["text/layers/0/mlp/gate"]["format"] == "q4_g64_fp16"
        # The draft path reduces the full output head through linear_topk, which only registers
        # Q8/FP8 at 248320 rows, so the head stays Q8.
        assert by_param["text/output_head"]["format"] == "q8_g32_fp16"
        # The offset norm is stored zero-centred, word for word.
        entry = by_param["text/layers/0/input_norm"]
        stored = np.frombuffer(
            artifact.read_object(entry["object"]), dtype="<u2"
        ).astype(np.uint32) << 16
        expected = (
            torch.from_numpy(tensors["blk.0.attn_norm.weight"] - 1.0)
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype(np.uint32)
            << 16
        )
        assert np.array_equal(stored, expected)
    assert report["objects"] > 20
