"""The writer side of a batched block-scale bank: one NVFP4 matrix and one divisor per expert."""

from __future__ import annotations

import json
import struct

from pathlib import Path

import pytest
import torch

from tools.artifact.layouts import (
    batched_block_scale_geometry,
    block_scale_geometry,
    encoded_size,
    get_layout,
)
from tools.artifact.reader import Artifact
from tools.artifact.schema import ArtifactError, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter

BANK = "batched_block_scale_k16_m128x4_v1"
EXPERTS, ROWS, COLUMNS = 2, 128, 128
CODE_BYTES = EXPERTS * ROWS * COLUMNS // 2
SCALE_OFFSET = CODE_BYTES
SCALE_PER_MATRIX = ROWS * (COLUMNS // 16)
DIVISOR_OFFSET = SCALE_OFFSET + EXPERTS * SCALE_PER_MATRIX
PAYLOAD_BYTES = DIVISOR_OFFSET + 4 * EXPERTS


def _codes() -> torch.Tensor:
    """One 128x128 code plane whose first two columns hold the E2M1 word 2 (1.0)."""

    codes = torch.zeros((ROWS, COLUMNS // 2), dtype=torch.uint8)
    codes[0, 0] = 0x22
    return codes


def _scales() -> torch.Tensor:
    """One natural scale plane whose first K group holds E4M3FN 0x40 (2.0)."""

    scales = torch.zeros((ROWS, COLUMNS // 16), dtype=torch.uint8)
    scales[0, 0] = 0x40
    return scales


def _bank(tmp_path: Path, divisors: int | None = None, words=None) -> Path:
    path = tmp_path / "bank.ninfer"
    words = [torch.tensor(2.0), torch.tensor(3.0)] if words is None else words
    with ArtifactWriter(
        path,
        [TensorSpec("bank", (EXPERTS, ROWS, COLUMNS), "nvfp4", BANK, divisors)],
        components={"text": {"config": {}}},
        bindings={"bank": {"object": "bank"}},
    ) as writer:
        output = TensorOutput(writer, "bank")
        for expert in range(EXPERTS):
            output.write_codes(expert * ROWS, _codes(), _scales())
        output.write_divisors(words)
        return path


def test_the_layout_repeats_one_matrix_and_one_divisor_per_expert():
    layout = get_layout(BANK)
    assert layout.name == BANK and layout.alignment == 256
    assert layout.formats == frozenset(("nvfp4",))
    geometry = batched_block_scale_geometry("nvfp4", (EXPERTS, ROWS, COLUMNS))
    assert (
        geometry.experts,
        geometry.n,
        geometry.k,
    ) == (EXPERTS, ROWS, COLUMNS)
    assert geometry.code_plane_bytes == CODE_BYTES
    assert geometry.scale_plane_offset == SCALE_OFFSET
    assert geometry.scale_bytes_per_matrix == SCALE_PER_MATRIX
    assert geometry.scale_plane_bytes == EXPERTS * SCALE_PER_MATRIX
    assert geometry.weight_divisor_offset == DIVISOR_OFFSET
    assert geometry.divisor_count == EXPERTS
    assert geometry.payload_bytes == PAYLOAD_BYTES
    assert (
        encoded_size(BANK, "nvfp4", (EXPERTS, ROWS, COLUMNS), divisors=EXPERTS)
        == PAYLOAD_BYTES
    )
    # The single-divisor parent keeps the one word it has always stored.
    assert block_scale_geometry("nvfp4", (ROWS, COLUMNS)).payload_bytes == 9220


def test_an_unstated_count_is_omitted_and_every_expert_keeps_its_word(tmp_path):
    path = _bank(tmp_path)
    raw = path.read_bytes()
    reserve = struct.unpack("<Q", raw[8:16])[0]
    directory = json.loads(raw[32 : 32 + reserve])
    record = directory["objects"][0]
    assert "divisors" not in record
    assert record["layout"] == BANK and record["shape"] == [EXPERTS, ROWS, COLUMNS]
    assert record["bytes"] == PAYLOAD_BYTES
    with Artifact(path) as artifact:
        assert artifact.object("bank").divisors is None
        assert artifact.object("bank").bytes == PAYLOAD_BYTES
        payload = artifact.read_object("bank")
    assert len(payload) == PAYLOAD_BYTES
    assert payload[:2] == b"\x22\x00" and payload[ROWS * COLUMNS // 2 :][:2] == b"\x22\x00"
    assert payload[SCALE_OFFSET] == 0x40
    assert payload[SCALE_OFFSET + SCALE_PER_MATRIX] == 0x40
    assert struct.unpack("<2f", payload[DIVISOR_OFFSET:]) == (2.0, 3.0)


def test_a_stated_count_is_written_and_must_match_the_expert_axis(tmp_path):
    path = _bank(tmp_path, divisors=EXPERTS)
    raw = path.read_bytes()
    reserve = struct.unpack("<Q", raw[8:16])[0]
    assert json.loads(raw[32 : 32 + reserve])["objects"][0]["divisors"] == EXPERTS
    with Artifact(path) as artifact:
        assert artifact.object("bank").divisors == EXPERTS
    with pytest.raises(ArtifactError, match="differs from the 2 expert matrices"):
        _bank(tmp_path, divisors=3)
    with pytest.raises(ArtifactError, match="differs from the 2 expert matrices"):
        _bank(tmp_path, divisors=1)
    with pytest.raises(ArtifactError, match="expected integer in"):
        _bank(tmp_path, divisors=0)


def test_a_single_divisor_layout_refuses_a_divisor_array(tmp_path):
    with pytest.raises(ArtifactError, match="stores a single matrix divisor"):
        with ArtifactWriter(
            tmp_path / "matrix.ninfer",
            [
                TensorSpec(
                    "matrix", (ROWS, COLUMNS), "nvfp4", "block_scale_k16_m128x4_v1", 2
                )
            ],
            components={"text": {"config": {}}},
            bindings={},
        ):
            pass


def test_the_divisor_array_must_be_one_valid_word_per_expert(tmp_path):
    with pytest.raises(ValueError, match="2 divisor domains, got 3 words"):
        _bank(tmp_path, words=[torch.tensor(2.0)] * 3)
    for word in (torch.tensor(0.0), torch.tensor(-1.0), struct.pack("<f", float("inf"))):
        with pytest.raises(ValueError, match="finite and positive"):
            _bank(tmp_path, words=[torch.tensor(2.0), word])


def test_a_batched_bank_does_not_take_a_divisor_word_per_row_block(tmp_path):
    path = tmp_path / "bank.ninfer"
    with pytest.raises(ValueError, match="divisors through write_divisors"):
        with ArtifactWriter(
            path,
            [TensorSpec("bank", (EXPERTS, ROWS, COLUMNS), "nvfp4", BANK)],
            components={"text": {"config": {}}},
            bindings={"bank": {"object": "bank"}},
        ) as writer:
            output = TensorOutput(writer, "bank")
            output.write_codes(0, _codes(), _scales(), struct.pack("<f", 2.0))
