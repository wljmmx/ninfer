"""GGUF input: container/dequant oracle, Qwen4Exp mapping, and a real conversion.

The reference for both layers is llama.cpp's own gguf-py: the tiny file below is written by
gguf-py's writer and its quantized blocks are decoded by gguf-py's dequantizers, so the reader
is checked against the implementation that defines the format rather than against itself. The
end-to-end arm then runs the real converter on that file and checks the artifact's contents
(the PLE hash controls byte for byte, a bf16 tensor word for word).

gguf-py is a test-only dependency here (the converter itself reads GGUF natively): the module
skips when it is not importable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gguf")
import gguf  # noqa: E402
from gguf.constants import GGMLQuantizationType as QT, GGML_QUANT_SIZES  # noqa: E402

from tools.convert.gguf_qwen4exp import GGUFSource  # noqa: E402
from tools.convert.sources.gguf import GGUFFile  # noqa: E402

HIDDEN, VOCAB, LAYERS = 64, 32, 2
HEADS, KV_HEADS, HEAD_DIM = 2, 1, 8
HC_COUNT, HC_RANK = 2, 4
NK, NV, DK, DV = 2, 4, 4, 4
CONV_K = 3
EXPERTS, INTER, SHARED = 4, 16, 8
IDX_HEADS, IDX_KV, IDX_DIM, IDX_TOPK = 2, 1, 4, 4
PLE_ED, NGRAM, HEADS_PER_NGRAM, PLE_CONV = 8, 3, 2, 2
Q = NK * DK
V = NV * DV
C = 2 * Q + V
QSA_Q = HEADS * HEAD_DIM
QSA_KV = KV_HEADS * HEAD_DIM
IQ = IDX_HEADS * IDX_DIM
IK = IDX_KV * IDX_DIM


def _t(rng, shape):
    return (rng.standard_normal(shape) * 0.05).astype(np.float16)


def _write_tiny(dir_path: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """A minimal Qwen4Exp GGUF: one linear layer, one QSA+PLE layer, MoE everywhere."""
    rng = np.random.default_rng(11)
    model = dir_path / "tiny.gguf"
    tensors: dict[str, np.ndarray] = {}

    def add(name, data):
        tensors[name] = data
        w.add_tensor(name, data)

    w = gguf.GGUFWriter(str(model), "qwen4exp")
    w.add_block_count(LAYERS)
    w.add_context_length(128)
    w.add_embedding_length(HIDDEN)
    w.add_feed_forward_length(SHARED)
    w.add_head_count(HEADS)
    w.add_head_count_kv(KV_HEADS)
    w.add_key_length(HEAD_DIM)
    w.add_value_length(HEAD_DIM)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(EXPERTS)
    w.add_expert_used_count(2)
    w.add_expert_feed_forward_length(INTER)
    w.add_expert_shared_feed_forward_length(SHARED)
    w.add_ssm_conv_kernel(CONV_K)
    w.add_ssm_state_size(DK)
    w.add_ssm_group_count(NK)
    w.add_ssm_time_step_rank(NV)
    w.add_ssm_inner_size(V)
    w.add_hyper_connection_count(HC_COUNT)
    w.add_hyper_connection_low_rank(HC_RANK)
    w.add_indexer_head_count(IDX_HEADS)
    w.add_indexer_key_length(IDX_DIM)
    w.add_indexer_top_k(IDX_TOPK)
    w.add_attention_compress_ratios([0, 8])  # layer 0 linear, layer 1 sparse QSA
    w.add_rope_freq_base(1000000.0)
    w.add_ple_layers([1])
    w.add_ple_ngram_size(NGRAM)
    w.add_ple_heads_per_ngram(HEADS_PER_NGRAM)
    w.add_ple_conv_kernel(PLE_CONV)
    w.add_ple_eos_token_id(1)
    w.add_embedding_length_per_layer_input(PLE_ED)
    # PLE hash controls are metadata arrays (u64), not tensors.
    w.add_ple_layer_multipliers([5, 6, 7])
    w.add_ple_head_offsets([0, 16, 32, 48])
    w.add_ple_head_vocab_sizes([64, 64, 32, 32])

    add("token_embd.weight", _t(rng, (VOCAB, HIDDEN)))
    add("output.weight", _t(rng, (VOCAB, HIDDEN)))
    add("output_hc_norm.weight", _t(rng, (HC_COUNT * HIDDEN,)))
    add("output_hc_down.weight", _t(rng, (HC_RANK, HC_COUNT * HIDDEN)))
    add("output_hc_up.weight", _t(rng, (HC_COUNT * HIDDEN, HC_RANK)))

    for layer in range(LAYERS):
        p = f"blk.{layer}."
        for side in ("attn", "ffn"):
            add(p + f"hc_{side}_norm.weight", _t(rng, (HC_COUNT * HIDDEN,)))
            add(p + f"hc_{side}_down_inject.weight",
                _t(rng, (HC_RANK + HC_COUNT, HC_COUNT * HIDDEN)))
            add(p + f"hc_{side}_up.weight", _t(rng, (HC_COUNT * HIDDEN, HC_RANK)))
        add(p + "ffn_gate_inp.weight", _t(rng, (EXPERTS, HIDDEN)))
        add(p + "ffn_gate_inp_shexp.weight", _t(rng, (1, HIDDEN)))
        add(p + "ffn_gate_shexp.weight", _t(rng, (SHARED, HIDDEN)))
        add(p + "ffn_up_shexp.weight", _t(rng, (SHARED, HIDDEN)))
        add(p + "ffn_down_shexp.weight", _t(rng, (HIDDEN, SHARED)))
        add(p + "ffn_gate_up_exps.weight", _t(rng, (EXPERTS, 2 * INTER, HIDDEN)))
        add(p + "ffn_down_exps.weight", _t(rng, (EXPERTS, HIDDEN, INTER)))
        if layer == 0:
            add(p + "attn_qkv_gate.weight", _t(rng, (C + V, HIDDEN)))
            add(p + "ssm_alpha.weight", _t(rng, (NV, HIDDEN)))
            add(p + "ssm_beta.weight", _t(rng, (NV, HIDDEN)))
            add(p + "ssm_conv1d.weight", _t(rng, (C, CONV_K)))
            add(p + "ssm_a.weight", _t(rng, (NV,)))
            add(p + "ssm_dt.weight", _t(rng, (NV,)))
            add(p + "ssm_norm.weight", _t(rng, (DV,)))
            add(p + "ssm_out.weight", _t(rng, (HIDDEN, V)))
        else:
            add(p + "attn_q.weight", _t(rng, (2 * QSA_Q, HIDDEN)))
            add(p + "attn_k.weight", _t(rng, (QSA_KV, HIDDEN)))
            add(p + "attn_v.weight", _t(rng, (QSA_KV, HIDDEN)))
            add(p + "attn_output.weight", _t(rng, (HIDDEN, QSA_Q)))
            add(p + "attn_q_norm.weight", _t(rng, (HEAD_DIM,)))
            add(p + "attn_k_norm.weight", _t(rng, (HEAD_DIM,)))
            add(p + "indexer.q_proj.weight", _t(rng, (IQ, HIDDEN)))
            add(p + "indexer.k_proj.weight", _t(rng, (IK, HIDDEN)))
            add(p + "indexer.q_norm.weight", _t(rng, (IDX_DIM,)))
            add(p + "indexer.k_norm.weight", _t(rng, (IDX_DIM,)))
            add(p + "ple_key.weight", _t(rng, (HIDDEN * HC_COUNT, PLE_ED)))
            add(p + "ple_value.weight", _t(rng, (HIDDEN, PLE_ED)))
            for role in ("norm_key", "norm_query", "norm_conv"):
                add(p + f"ple_{role}.weight", _t(rng, (HIDDEN * HC_COUNT,)))
            add(p + "ple_conv1d.weight", _t(rng, (HIDDEN * HC_COUNT, PLE_CONV)))
            add("per_layer_token_embd.weight", _t(rng, (16, PLE_ED)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    (dir_path / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": {f"t{i}": i for i in range(VOCAB)}, "merges": []},
        "added_tokens": [],
    }))
    (dir_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128}))
    (dir_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    return model, tensors


def test_container_and_dequantizers_match_gguf_py(tmp_path):
    """The reader reproduces gguf-py's dequantization word for word, including bounded reads."""
    rng = np.random.default_rng(3)
    model = tmp_path / "blocks.gguf"
    w = gguf.GGUFWriter(str(model), "qwen4exp")
    cases = {}
    for name, qtype, shape in (
        ("q8_0", QT.Q8_0, (8, 256)),
        ("q4_k", QT.Q4_K, (8, 256)),
        ("q6_k", QT.Q6_K, (4, 256)),
        ("q2_k", QT.Q2_K, (4, 256)),
        ("q5_k", QT.Q5_K, (4, 256)),
        ("q4_0", QT.Q4_0, (8, 256)),
    ):
        block_size, type_size = GGML_QUANT_SIZES[qtype]
        n_bytes = shape[0] * shape[1] // block_size * type_size
        raw = rng.integers(0, 256, size=n_bytes, dtype=np.uint8)
        byte_shape = (shape[0], n_bytes // shape[0])
        w.add_tensor(name, raw.reshape(byte_shape), raw_shape=byte_shape, raw_dtype=qtype)
        cases[name] = (qtype, raw, shape)
    values = rng.standard_normal((8, 256)).astype(np.float16)
    w.add_tensor("f16", values)
    cases["f16"] = (QT.F16, values.view(np.uint16).reshape(-1), (8, 256))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    with GGUFFile(model) as file:
        for name, (qtype, raw, shape) in cases.items():
            assert file.tensors[name].shape == shape
            got = file.read_flat(name)
            ref = gguf.quants.dequantize(raw, qtype).reshape(-1)
            assert np.array_equal(got, ref, equal_nan=True), name
        # A bounded range must be the same elements, not a re-read of the whole tensor.
        part = file.read_flat("q4_k", 100, 400)
        ref = gguf.quants.dequantize(cases["q4_k"][1], QT.Q4_K).reshape(-1)[100:400]
        assert np.array_equal(part, ref, equal_nan=True)


def test_source_maps_qwen4exp_names(tmp_path):
    model, tensors = _write_tiny(tmp_path)
    with GGUFSource(model) as source:
        config = source.config["text_config"]
        # HF's own spelling; qwen4_exp.text_config maps full_attention -> indexed_attention.
        assert config["layer_types"] == ["linear_attention", "full_attention"]
        assert config["ple_layer_ids"] == [2]  # HF's 1-based ids (the GGUF stores 0-based)
        assert config["linear_value_head_dim"] == DV
        assert config["num_experts_per_tok"] == 2
        names = source.weight_map
        # The builder asks for these exact HF names; a missing one is a refused conversion.
        for required in (
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            "model.language_model.layers.0.mlp.experts.down_proj",
            "model.language_model.layers.1.self_attn.q_proj.weight",
            "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight",
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
        ):
            assert required in names, required
        # Shapes follow the builder's contract, not the GGUF spelling.
        assert source.describe(
            "model.language_model.layers.0.linear_attn.conv1d.weight").shape == (C, 1, CONV_K)
        assert source.describe(
            "model.language_model.layers.0.mlp.experts.gate_up_proj").shape == (
                EXPERTS, 2 * INTER, HIDDEN)
        assert source.describe(
            "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight").shape == (
                IQ + IK, HIDDEN)
        assert source.describe(
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers").dtype == "I64"


def test_merged_tensors_are_split_back(tmp_path):
    torch = pytest.importorskip("torch")
    model, tensors = _write_tiny(tmp_path)
    with GGUFSource(model) as source:
        # The QSA q|gate parent stays one interleaved (2Q, H) tensor.
        q = source.read_flat("model.language_model.layers.1.self_attn.q_proj.weight").reshape(
            2 * QSA_Q, HIDDEN)
        assert torch.equal(
            q,
            torch.from_numpy(tensors["blk.1.attn_q.weight"].astype(np.float32)).to(torch.bfloat16),
        )
        # The indexer projection is fused back as [q; k].
        fused = source.read_flat(
            "model.language_model.layers.1.self_attn.indexer.index_qk_proj.weight").reshape(
                IQ + IK, HIDDEN)
        iq = torch.from_numpy(
            tensors["blk.1.indexer.q_proj.weight"].astype(np.float32)).to(torch.bfloat16)
        ik = torch.from_numpy(
            tensors["blk.1.indexer.k_proj.weight"].astype(np.float32)).to(torch.bfloat16)
        assert torch.equal(fused, torch.cat([iq, ik], dim=0))
        # The linear-attention qkv|z pair splits at C. The V sections (and z) are in llama.cpp's
        # tiled order on disk and must come back in HF's grouped order, so the two halves are
        # checked separately: q|k is copied, V is permuted, exactly once.
        qkv = source.read_flat(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight").reshape(C, HIDDEN)
        z = source.read_flat(
            "model.language_model.layers.0.linear_attn.in_proj_z.weight").reshape(V, HIDDEN)
        merged = torch.from_numpy(
            tensors["blk.0.attn_qkv_gate.weight"].astype(np.float32)).to(torch.bfloat16)
        from tools.convert.gguf_qwen4exp import tiled_to_grouped

        perm = tiled_to_grouped(V, NK, NV // NK, DV)
        # Head-level map (head_dim=1): [v0 of k0, v0 of k1, v1 of k0, v1 of k1].
        assert tiled_to_grouped(NV, NK, NV // NK, 1).tolist() == [0, 2, 1, 3]
        assert torch.equal(qkv[: 2 * Q], merged[: 2 * Q])
        assert torch.equal(qkv[2 * Q :], merged[2 * Q :][torch.from_numpy(perm)])
        assert torch.equal(z, merged[C:][torch.from_numpy(perm)])


def test_conversion_from_gguf(tmp_path, monkeypatch):
    """The real converter reads the GGUF and writes a v3 artifact whose contents are checked."""
    torch = pytest.importorskip("torch")
    from tools.convert.__main__ import main as convert_main
    from tools.artifact.reader import Artifact

    model, tensors = _write_tiny(tmp_path)
    out = tmp_path / "tiny.ninfer"
    convert_main([
        "--model", str(model),
        "--recipe", "qwen4_exp",
        "--components", "text",
        "--device", "cpu",
        "--name", "tiny-qwen4exp",
        "--out", str(out),
    ])
    assert out.is_file()
    report = json.loads(Path(str(out) + ".conversion.json").read_text())
    by_param = {}
    for method in report["methods"]:
        for name in method["parameters"]:
            by_param[name] = method

    with Artifact.open(out) as artifact:
        # The PLE hash controls are metadata arrays: they must survive as exact int64 bytes,
        # in the builder's concatenation order (multipliers, vocab sizes, offsets).
        expected = np.array([5, 6, 7, 64, 64, 32, 32, 0, 16, 32, 48], dtype="<i8").tobytes()
        assert artifact.read_object("resource/text/ple_hash_controls") == expected
        # A bf16 parameter is stored word for word (the source f16 is widened then rounded to
        # bf16 by the source layer, which is what the converter writes).
        norm = by_param["text/layers/0/gdn/norm"]
        stored = np.frombuffer(artifact.read_object(norm["object"]), dtype="<u2")
        source_words = (
            torch.from_numpy(tensors["blk.0.ssm_norm.weight"].astype(np.float32))
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
        )
        assert np.array_equal(stored, source_words)
    assert report["objects"] > 50
