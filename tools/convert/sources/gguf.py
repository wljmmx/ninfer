"""GGUF container, metadata and quantized tensor reader (numpy only).

Why this exists: the Qwen4Exp path can arrive as a GGUF when only a quantized distribution
exists. NInfer's converter reads *float* values from a source and re-quantizes them into its own
formats, so it needs one thing from a GGUF: the tensor values, dequantized, under their logical
(HF) names. That is what this module provides, and it is deliberately torch-free - the container
and every dequantizer are numpy, so they can be checked without the conversion stack installed
(see tests/convert/test_gguf.py, which cross-checks against gguf-py's own writer/dequantizer).

Quantization code is a port of gguf-py's numpy dequantizers (llama.cpp, MIT). The block layouts
are the format, not an implementation detail: F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, IQ4_NL,
IQ4_XS and the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K. Anything else is refused by name instead of
being guessed at.

GGUF stores the fastest dimension first (`ne[0]`), while numpy/torch are row-major (last axis
fastest), so a stored shape is reversed on the way in. Elements keep their original flat order:
reading a bounded element range [begin, end) maps 1:1 onto the stored byte stream, which is what
lets a 27 GiB embedding table be paged in by block range instead of materialized whole.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import prod
import os
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np

from tools.artifact.file_io import pread

GGUF_MAGIC = 0x46554747
_GGUF_MAX_VERSION = 3

# GGUF metadata value types.
_GGUF_UINT8, _GGUF_INT8, _GGUF_UINT16, _GGUF_INT16, _GGUF_UINT32, _GGUF_INT32 = range(6)
_GGUF_FLOAT32, _GGUF_BOOL, _GGUF_STRING, _GGUF_ARRAY, _GGUF_UINT64, _GGUF_INT64 = range(6, 12)
_GGUF_FLOAT64 = 12

# ggml tensor types. Only the layouts this reader can dequantize exactly are listed; a file that
# uses another one is refused by name (the alternative is silently wrong weights).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
GGML_IQ4_NL = 20
GGML_IQ4_XS = 23
GGML_I8 = 24
GGML_I16 = 25
GGML_I32 = 26
GGML_I64 = 27
GGML_F64 = 28
GGML_BF16 = 30

QK_K = 256

# type -> (block_size, bytes_per_block)
QUANT_SIZES: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (QK_K, 84),
    GGML_Q3_K: (QK_K, 110),
    GGML_Q4_K: (QK_K, 144),
    GGML_Q5_K: (QK_K, 176),
    GGML_Q6_K: (QK_K, 210),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ4_XS: (QK_K, 136),
    GGML_I8: (1, 1),
    GGML_I16: (1, 2),
    GGML_I32: (1, 4),
    GGML_I64: (1, 8),
    GGML_F64: (1, 8),
    GGML_BF16: (1, 2),
}

QUANT_NAMES = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_I8: "I8",
    GGML_I16: "I16",
    GGML_I32: "I32",
    GGML_I64: "I64",
    GGML_F64: "F64",
    GGML_BF16: "BF16",
}


# --------------------------------------------------------------------- dequantizers (gguf-py port)
# Each function takes (n_blocks, type_size) uint8 rows and returns (n_blocks, block_size) float32
# rows - gguf-py's own contract, kept so the port stays a mechanical translation.


def _q2_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    scales, rest = np.hsplit(blocks, [QK_K // 16])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    d, dmin = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    dl = (d * (scales & 0xF).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
    shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & np.uint8(3)
    qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)
    return (dl * qs - ml).reshape((n_blocks, -1))


def _q3_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    hmask, rest = np.hsplit(blocks, [QK_K // 8])
    qs, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    lscales, hscales = np.hsplit(scales, [8])
    lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & np.uint8(0x0F)) | ((hscales & np.uint8(0x03)) << np.uint8(4))
    scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1)) ^ np.uint8(1)
    q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
    return (dl * q).reshape((n_blocks, QK_K))


def _q4_k_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = scales.shape[0]
    scales = scales.view(np.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def _q4_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, qs = np.hsplit(rest, [12])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 32)).astype(np.float32)
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _q5_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    dmin, rest = np.hsplit(rest, [2])
    scales, rest = np.hsplit(rest, [12])
    qh, qs = np.hsplit(rest, [QK_K // 8])
    d = d.view(np.float16).astype(np.float32)
    dmin = dmin.view(np.float16).astype(np.float32)
    sc, m = _q4_k_scale_min(scales)
    d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = (qh & np.uint8(0x01)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.float32)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _q6_k(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    ql, rest = np.hsplit(blocks, [QK_K // 2])
    qh, rest = np.hsplit(rest, [QK_K // 4])
    scales, d = np.hsplit(rest, [QK_K // 16])
    scales = scales.view(np.int8).astype(np.float32)
    d = d.view(np.float16).astype(np.float32)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = (qh & np.uint8(0x03)).reshape((n_blocks, -1, 32))
    q = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(32)
    q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)
    return (d * q).reshape((n_blocks, QK_K))


def _q4_0(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)
    return d * qs.astype(np.float32)


def _q4_1(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    m, qs = np.hsplit(rest, [2])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1)).astype(np.float32)
    return d * qs + m


def _q5_0(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32).reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.int8) - np.int8(16)
    return d * qs.astype(np.float32)


def _q5_1(blocks: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]
    d, rest = np.hsplit(blocks, [2])
    m, rest = np.hsplit(rest, [2])
    qh, qs = np.hsplit(rest, [4])
    d = d.view(np.float16).astype(np.float32)
    m = m.view(np.float16).astype(np.float32)
    qh = qh.view(np.uint32).reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
    ql = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((n_blocks, -1))
    qs = (ql | (qh << np.uint8(4))).astype(np.float32)
    return d * qs + m


def _q8_0(blocks: np.ndarray) -> np.ndarray:
    d, x = np.split(blocks, [2], axis=1)
    d = d.view(np.float16).astype(np.float32)
    x = x.view(np.int8).astype(np.float32)
    return x * d


_IQ4_NL_VALUES = np.array(
    (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113), dtype=np.int8
)


def _iq4_nl(blocks: np.ndarray) -> np.ndarray:
    """4-bit non-linear codes: one fp16 scale plus 16 nibbles per 32 values (gguf-py layout)."""
    n_blocks = blocks.shape[0]
    d, qs = np.hsplit(blocks, [2])
    d = d.view(np.float16).astype(np.float32)
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qs = (qs & np.uint8(0x0F)).reshape((n_blocks, -1, 1))
    values = _IQ4_NL_VALUES.reshape((1, 1, 16))
    qs = np.take_along_axis(values, qs.astype(np.int64), axis=-1).astype(np.float32).reshape(
        (n_blocks, -1)
    )
    return d * qs


def _iq4_xs(blocks: np.ndarray) -> np.ndarray:
    """4-bit non-linear codes with 6-bit sub-scales: one fp16 super-scale, then per 32 values a
    6-bit signed scale and 16 packed nibbles (ggml block_iq4_xs: d, scales_h, scales_l[4],
    qs[128]).

    The reference is llama.cpp's dequantize_row_iq4_xs; group ib's sub-scale is
    ``((scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((scales_h >> 2*ib) & 3) << 4)``, biased by 32.
    """
    n_blocks = blocks.shape[0]
    d = blocks[:, 0:2].view(np.float16).astype(np.float32)
    scales_h = blocks[:, 2:4].view(np.uint16).astype(np.int32)
    scales_l = blocks[:, 4:8].astype(np.int32)
    qs = blocks[:, 8:136].reshape(n_blocks, 8, 16)
    values = _IQ4_NL_VALUES.reshape((1, 1, 1, 16))
    low = np.take_along_axis(values, (qs & 0x0F).astype(np.int64)[..., None], axis=-1)[..., 0]
    high = np.take_along_axis(values, (qs >> 4).astype(np.int64)[..., None], axis=-1)[..., 0]
    # The reference emits y[j] from the low nibble and y[j+16] from the high one, so a 32-value
    # group is the 16 low values followed by the 16 high ones.
    pair = np.concatenate([low, high], axis=-1).astype(np.float32)
    shifts = np.array([0, 4, 0, 4, 0, 4, 0, 4], dtype=np.int32)
    scale = ((scales_l[:, [0, 0, 1, 1, 2, 2, 3, 3]] >> shifts) & 0xF) | (
        ((scales_h >> (2 * np.arange(8, dtype=np.int32))) & 3) << 4
    )
    dl = d * (scale - 32).astype(np.float32)
    return (pair * dl[:, :, None]).reshape(n_blocks, 256)


_DEQUANT = {
    GGML_Q4_0: _q4_0,
    GGML_Q4_1: _q4_1,
    GGML_Q5_0: _q5_0,
    GGML_Q5_1: _q5_1,
    GGML_Q8_0: _q8_0,
    GGML_IQ4_NL: _iq4_nl,
    GGML_IQ4_XS: _iq4_xs,
    GGML_Q2_K: _q2_k,
    GGML_Q3_K: _q3_k,
    GGML_Q4_K: _q4_k,
    GGML_Q5_K: _q5_k,
    GGML_Q6_K: _q6_k,
}


def dequantize(blocks: np.ndarray, ggml_type: int) -> np.ndarray:
    """(n_blocks, bytes_per_block) uint8 -> (n_blocks, block_size) float32."""
    if ggml_type == GGML_F32:
        return blocks.view(np.float32)
    if ggml_type == GGML_F16:
        return blocks.view(np.float16).astype(np.float32)
    if ggml_type == GGML_BF16:
        return (blocks.view(np.int16).astype(np.int32) << 16).view(np.float32)
    if ggml_type == GGML_F64:
        return blocks.view(np.float64).astype(np.float32)
    if ggml_type == GGML_I8:
        return blocks.view(np.int8).astype(np.float32)
    if ggml_type == GGML_I16:
        return blocks.view(np.int16).astype(np.float32)
    if ggml_type == GGML_I32:
        return blocks.view(np.int32).astype(np.float32)
    if ggml_type == GGML_I64:
        return blocks.view(np.int64).astype(np.float32)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise ValueError(
            f"GGUF tensor type {QUANT_NAMES.get(ggml_type, ggml_type)} is not supported; "
            "re-quantize the source (F16/BF16/Q8_0 and the K-quants are readable)"
        )
    return fn(blocks)


# ---------------------------------------------------------------------------------- container


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True, slots=True)
class GGUFTensor:
    name: str
    shape: tuple[int, ...]  # logical (row-major) shape
    ggml_type: int
    offset: int  # absolute file offset of the first byte
    nbytes: int

    @property
    def dtype_name(self) -> str:
        return QUANT_NAMES.get(self.ggml_type, str(self.ggml_type))


class _Cursor:
    """Little-endian sequential reader over the open file's header region."""

    def __init__(self, stream) -> None:
        self.stream = stream

    def take(self, count: int) -> bytes:
        data = self.stream.read(count)
        if len(data) != count:
            raise ValueError("truncated GGUF header")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        if length > 1 << 30:
            raise ValueError("GGUF string length is implausible")
        return self.take(length).decode("utf-8", "replace")


def _read_value(cur: _Cursor) -> Any:
    kind = cur.u32()
    if kind == _GGUF_STRING:
        return cur.string()
    if kind == _GGUF_ARRAY:
        item_type = cur.u32()
        count = cur.u64()
        if count > 1 << 32:
            raise ValueError("GGUF array length is implausible")
        return np.array([_read_typed(cur, item_type) for _ in range(count)]) if count else []
    return _read_typed(cur, kind)


def _read_typed(cur: _Cursor, kind: int) -> Any:
    if kind == _GGUF_UINT8:
        return struct.unpack("<B", cur.take(1))[0]
    if kind == _GGUF_INT8:
        return struct.unpack("<b", cur.take(1))[0]
    if kind == _GGUF_UINT16:
        return struct.unpack("<H", cur.take(2))[0]
    if kind == _GGUF_INT16:
        return struct.unpack("<h", cur.take(2))[0]
    if kind == _GGUF_UINT32:
        return cur.u32()
    if kind == _GGUF_INT32:
        return struct.unpack("<i", cur.take(4))[0]
    if kind == _GGUF_FLOAT32:
        return struct.unpack("<f", cur.take(4))[0]
    if kind == _GGUF_BOOL:
        return bool(struct.unpack("<B", cur.take(1))[0])
    if kind == _GGUF_STRING:
        return cur.string()
    if kind == _GGUF_UINT64:
        return cur.u64()
    if kind == _GGUF_INT64:
        return struct.unpack("<q", cur.take(8))[0]
    if kind == _GGUF_FLOAT64:
        return struct.unpack("<d", cur.take(8))[0]
    if kind == _GGUF_ARRAY:
        raise ValueError("nested GGUF arrays are not defined")
    raise ValueError(f"unknown GGUF metadata value type {kind}")


class GGUFFile:
    """One GGUF file: metadata, tensor directory, and bounded dequantized reads.

    The directory is parsed once (a few hundred entries); tensor data is read with pread on
    demand, one block-aligned window at a time, so a 27 GiB table never has to be resident.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root = self.path.parent
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensor] = {}
        self.alignment = 32
        self._fd: int | None = None
        self.bytes_read = 0
        self._parse()

    # -- header ---------------------------------------------------------
    def _parse(self) -> None:
        file_bytes = self.path.stat().st_size
        with self.path.open("rb") as stream:
            cur = _Cursor(stream)
            head = cur.take(24)
            magic, version, tensor_count, kv_count = struct.unpack("<IIQQ", head)
            if magic != GGUF_MAGIC:
                raise ValueError(f"{self.path}: not a GGUF file (magic {magic:#x})")
            if version > _GGUF_MAX_VERSION:
                raise ValueError(f"{self.path}: GGUF version {version} is newer than this reader")
            for _ in range(kv_count):
                key = cur.string()
                self.metadata[key] = _read_value(cur)
            entries: list[tuple[str, tuple[int, ...], int, int]] = []
            for _ in range(tensor_count):
                name = cur.string()
                n_dims = cur.u32()
                if n_dims == 0 or n_dims > 8:
                    raise ValueError(f"{self.path}/{name}: invalid dimension count {n_dims}")
                dims = [cur.u64() for _ in range(n_dims)]
                ggml_type = cur.u32()
                rel_offset = cur.u64()
                entries.append((name, tuple(reversed(dims)), ggml_type, rel_offset))
            alignment = int(self.metadata.get("general.alignment", 32) or 32)
            data_start = _align(stream.tell(), alignment)
        if data_start > file_bytes and entries:
            raise ValueError(f"{self.path}: GGUF tensor data starts beyond the file")
        self.alignment = alignment
        tensors: dict[str, GGUFTensor] = {}
        for name, shape, ggml_type, rel_offset in entries:
            block, type_size = _sizes(ggml_type)
            elements = prod(shape)
            if block != 1 and elements % block != 0:
                raise ValueError(
                    f"{self.path}/{name}: {QUANT_NAMES.get(ggml_type)} needs a multiple of "
                    f"{block} elements, tensor has {elements}"
                )
            nbytes = elements // block * type_size
            offset = data_start + rel_offset
            if offset + nbytes > file_bytes:
                raise ValueError(f"{self.path}/{name}: tensor range exceeds the file")
            tensors[name] = GGUFTensor(name, shape, ggml_type, offset, nbytes)
        self.tensors = tensors

    # -- metadata -------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.metadata:
            raise ValueError(f"{self.path}: GGUF metadata is missing {key!r}")
        return self.metadata[key]

    def arch(self) -> str:
        return str(self.require("general.architecture"))

    # -- tensor data ----------------------------------------------------
    def _file(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def _read_bytes(self, offset: int, count: int) -> bytes:
        raw = pread(self._file(), count, offset)
        if len(raw) != count:
            raise ValueError(f"{self.path}: short read at {offset} ({count} bytes)")
        self.bytes_read += count
        return raw

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> np.ndarray:
        """Elements [begin, end) of one tensor, dequantized, in the stored element order.

        Integer tensors (I8/I16/I32/I64) come back in their own dtype - the PLE hash constants
        must stay exact - and every other tensor comes back as float32.
        """
        info = self.tensors.get(name)
        if info is None:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}")
        elements = prod(info.shape)
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: range [{begin},{end}) exceeds {info.shape}")
        if begin == end:
            dtype = _element_dtype(info.ggml_type)
            return np.empty(0, dtype=dtype)
        block, type_size = _sizes(info.ggml_type)
        first_block = begin // block
        last_block = (end - 1) // block + 1
        raw = self._read_bytes(info.offset + first_block * type_size,
                               (last_block - first_block) * type_size)
        if block == 1:
            native = _read_native(raw, info.ggml_type)
            low = begin - first_block  # == 0: a block is one element
            return native[low : low + (end - begin)]
        grid = np.frombuffer(raw, dtype=np.uint8).reshape((last_block - first_block, type_size))
        values = dequantize(grid, info.ggml_type).reshape(-1)
        return values[begin - first_block * block : end - first_block * block]

    def read_tensor(self, name: str) -> np.ndarray:
        info = self.tensors.get(name)
        if info is None:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}")
        return self.read_flat(name, 0, prod(info.shape)).reshape(info.shape)

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GGUFSet:
    """All parts of a split GGUF, presented as one source.

    llama.cpp's split writer emits `name-0000N-of-0000M.gguf`; the metadata (including
    `general.architecture`) lives in the part that carries it - the first one in practice, which
    may hold no tensors at all - while each part keeps its own tensor directory and data offsets.
    Reading is delegated per tensor, so a 100 GiB model is never materialized.
    """

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError("a GGUF set needs at least one part")
        self.parts = [GGUFFile(path) for path in paths]
        header = next(
            (part for part in self.parts if "general.architecture" in part.metadata),
            self.parts[0],
        )
        self.metadata = header.metadata
        self.alignment = header.alignment
        self.path = header.path
        self.root = header.root
        self.tensors: dict[str, GGUFTensor] = {}
        self._owner: dict[str, GGUFFile] = {}
        for part in self.parts:
            for name, info in part.tensors.items():
                if name in self.tensors:
                    raise ValueError(f"tensor {name!r} appears in more than one GGUF part")
                self.tensors[name] = info
                self._owner[name] = part

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.metadata:
            raise ValueError(f"{self.path}: GGUF metadata is missing {key!r}")
        return self.metadata[key]

    def arch(self) -> str:
        return str(self.require("general.architecture"))

    def read_flat(self, name: str, begin: int = 0, end: int | None = None) -> np.ndarray:
        owner = self._owner.get(name)
        if owner is None:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}")
        return owner.read_flat(name, begin, end)

    def read_tensor(self, name: str) -> np.ndarray:
        info = self.tensors.get(name)
        if info is None:
            raise ValueError(f"{self.path}: missing GGUF tensor {name!r}")
        return self.read_flat(name, 0, prod(info.shape)).reshape(info.shape)

    @property
    def bytes_read(self) -> int:
        return sum(part.bytes_read for part in self.parts)

    def close(self) -> None:
        for part in self.parts:
            part.close()

    def __enter__(self) -> "GGUFSet":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


_SPLIT_NAME = re.compile(r"^(?P<base>.*)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)


def open_gguf(path: str | Path) -> "GGUFFile | GGUFSet":
    """Open one GGUF, or every part of a split set when the name says it is one."""
    path = Path(path)
    match = _SPLIT_NAME.match(path.name)
    if match is None:
        return GGUFFile(path)
    total = int(match.group("total"))
    base = match.group("base")
    parts = [path.with_name(f"{base}-{index:05d}-of-{total:05d}.gguf") for index in range(1, total + 1)]
    missing = [part.name for part in parts if not part.exists()]
    if missing:
        raise ValueError(
            f"{path}: split GGUF is missing part(s) {missing}; all {total} parts are required"
        )
    return GGUFSet(parts)


def _sizes(ggml_type: int) -> tuple[int, int]:
    try:
        return QUANT_SIZES[ggml_type]
    except KeyError as error:
        raise ValueError(
            f"GGUF tensor type {QUANT_NAMES.get(ggml_type, ggml_type)} is not supported"
        ) from error


def _read_native(raw: bytes, ggml_type: int) -> np.ndarray:
    """A block==1 tensor's elements in their own dtype (integers stay exact).

    Float types are widened to float32 here: BF16 needs the explicit shift (a raw uint16 view
    would be the bit pattern, not the number), and F16's widening is lossless.
    """
    if ggml_type == GGML_I8:
        return np.frombuffer(raw, dtype=np.int8)
    if ggml_type == GGML_I16:
        return np.frombuffer(raw, dtype=np.int16)
    if ggml_type == GGML_I32:
        return np.frombuffer(raw, dtype=np.int32)
    if ggml_type == GGML_I64:
        return np.frombuffer(raw, dtype=np.int64)
    if ggml_type == GGML_F32:
        return np.frombuffer(raw, dtype=np.float32)
    if ggml_type == GGML_F64:
        return np.frombuffer(raw, dtype=np.float64).astype(np.float32)
    if ggml_type == GGML_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if ggml_type == GGML_BF16:
        return (np.frombuffer(raw, dtype=np.int16).astype(np.int32) << 16).view(np.float32)
    raise ValueError(f"unknown non-blocked GGUF type {ggml_type}")
