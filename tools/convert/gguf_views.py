"""Logical views over a GGUF tensor directory, shared by every GGUF source.

A view reads a flat element range ``[begin, end)`` and returns numpy values in the order the
converter expects. Integer views keep their dtype: 45-bit hash constants must reach the artifact
exact.

Moved out of the Qwen4Exp adapter when the Qwen3.5 dense adapter arrived; both present a GGUF as
the duck-typed source the architecture builders already consume, and both need the same range
views. Three transforms answer an encoding whose convention differs from the checkpoint's own:

* ``_Minus1`` - llama.cpp writes offset RMSNorm weights (``w + 1``), the artifact stores ``w``.
* ``_LogOfNegative`` - GDN keeps ``A_log``; llama.cpp writes ``ssm_a = -exp(A_log)``.
* ``_Transpose`` - the GDN conv kernel is ``[channels, taps]``, HF keeps ``[channels, 1, taps]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources.gguf import GGML_I64, GGUFFile


@dataclass(frozen=True)
class _Info:
    shape: tuple[int, ...]
    dtype: str  # "BF16" | "I64"
    offset: int = 0
    bytes: int = 0


class View:
    shape: tuple[int, ...]
    dtype: str

    def read(self, begin: int, end: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> _Info:
        elements = int(np.prod(self.shape)) if self.shape else 0
        width = 8 if self.dtype == "I64" else 2
        return _Info(tuple(self.shape), self.dtype, 0, elements * width)


def _empty(dtype: str) -> np.ndarray:
    return np.empty(0, dtype=np.int64 if dtype == "I64" else np.float32)


class _Direct(View):
    def __init__(self, file: GGUFFile, name: str) -> None:
        self.file = file
        self.name = name
        info = file.tensors[name]
        self.shape = info.shape
        self.dtype = "I64" if info.ggml_type == GGML_I64 else "BF16"

    def read(self, begin: int, end: int) -> np.ndarray:
        return self.file.read_flat(self.name, begin, end)


class _ArrayView(View):
    """A materialized block, used where a permutation needs a whole row section at once."""

    def __init__(self, values: np.ndarray, dtype: str = "BF16") -> None:
        self.values = np.ascontiguousarray(values, dtype=np.int64 if dtype == "I64" else np.float32)
        self.shape = tuple(self.values.shape)
        self.dtype = dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        return self.values.reshape(-1)[begin:end]


class _Rows(View):
    """Rows [start, start+rows) of a matrix."""

    def __init__(self, inner: View, start: int, rows: int, cols: int) -> None:
        self.inner, self.start, self.cols = inner, start, cols
        self.shape = (rows, cols)
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        if begin == end:
            return _empty(self.dtype)
        first = self.start + begin // self.cols
        last = self.start + (end - 1) // self.cols + 1
        block = self.inner.read(first * self.cols, last * self.cols)
        low = begin - (begin // self.cols) * self.cols
        return block[low : low + (end - begin)]


class _Cols(View):
    """Columns [start, start+cols) of a matrix, row by row."""

    def __init__(self, inner: View, start: int, cols: int, total: int | None = None) -> None:
        if len(inner.shape) != 2:
            raise ValueError("column selection requires a matrix")
        self.inner = inner
        self.start = start
        self.cols = cols
        self.total = int(total if total is not None else inner.shape[1])
        if start < 0 or cols <= 0 or start + cols > self.total:
            raise ValueError("column range is outside the matrix")
        self.shape = (int(inner.shape[0]), cols)
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        if begin == end:
            return _empty(self.dtype)
        first = begin // self.cols
        last = (end - 1) // self.cols + 1
        rows = self.inner.read(first * self.total, last * self.total).reshape(
            last - first, self.total
        )
        block = rows[:, self.start : self.start + self.cols].reshape(-1)
        low = begin - first * self.cols
        return block[low : low + (end - begin)]


class _ConcatRows(View):
    """Row-wise concatenation of views; each part contributes whole rows."""

    def __init__(self, parts: Iterable[tuple[View, int, int]]) -> None:
        self.parts = [(view, start, rows) for view, start, rows in parts]
        cols = {int(view.shape[1]) for view, _, _ in self.parts}
        if len(cols) != 1:
            raise ValueError("row concatenation requires a common column count")
        self.cols = cols.pop()
        self.shape = (sum(rows for _, _, rows in self.parts), self.cols)
        self.dtype = self.parts[0][0].dtype if self.parts else "BF16"

    def read(self, begin: int, end: int) -> np.ndarray:
        out: list[np.ndarray] = []
        cursor = 0
        for view, start, rows in self.parts:
            span = rows * self.cols
            low, high = max(begin, cursor), min(end, cursor + span)
            if low < high:
                out.append(view.read(start * self.cols + low - cursor,
                                     start * self.cols + high - cursor))
            cursor += span
            if cursor >= end:
                break
        if not out:
            return _empty(self.dtype)
        return np.concatenate(out)


class _FlatPermute(View):
    """1-D permutation (A_log, dt_bias and the beta/alpha rows)."""

    def __init__(self, inner: View, perm: np.ndarray) -> None:
        if len(inner.shape) != 1:
            raise ValueError("flat permutation requires a vector")
        self.inner = inner
        self.perm = np.asarray(perm, dtype=np.int64)
        self.shape = (len(self.perm),)
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        if begin == end:
            return _empty(self.dtype)
        values = np.concatenate(
            [self.inner.read(int(i), int(i) + 1) for i in self.perm[begin:end]]
        )
        return values


class _RowPermute(View):
    """Output row i is inner row perm[i]."""

    def __init__(self, inner: View, perm: np.ndarray) -> None:
        self.inner = inner
        self.perm = np.asarray(perm, dtype=np.int64)
        self.cols = int(inner.shape[1])
        self.shape = (len(self.perm), self.cols)
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        if begin == end:
            return _empty(self.dtype)
        first = begin // self.cols
        last = (end - 1) // self.cols + 1
        rows = np.concatenate(
            [
                self.inner.read(int(row) * self.cols, (int(row) + 1) * self.cols)
                for row in self.perm[first:last]
            ]
        )
        low = begin - first * self.cols
        return rows[low : low + (end - begin)]


class _ColPermute(View):
    """Output column j is inner column perm[j]."""

    def __init__(self, inner: View, perm: np.ndarray) -> None:
        self.inner = inner
        self.perm = np.asarray(perm, dtype=np.int64)
        self.cols = int(inner.shape[1])
        self.shape = inner.shape
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        if begin == end:
            return _empty(self.dtype)
        first = begin // self.cols
        last = (end - 1) // self.cols + 1
        block = self.inner.read(first * self.cols, last * self.cols).reshape(
            last - first, self.cols
        )[:, self.perm]
        flat = block.reshape(-1)
        low = begin - first * self.cols
        return flat[low : low + (end - begin)]


class _FusedExperts(View):
    """[E, 2I, H] with gate rows then up rows per expert, from one fused bank or two halves."""

    def __init__(self, fused: View | None, gate: View | None, up: View | None,
                 experts: int, width: int, hidden: int) -> None:
        if fused is None and (gate is None or up is None):
            raise ValueError("fused experts need either the fused bank or both halves")
        self.fused, self.gate, self.up = fused, gate, up
        self.experts, self.width, self.hidden = experts, width, hidden
        self.shape = (experts, 2 * width, hidden)
        self.dtype = "BF16"

    def _half(self, kind: str, expert: int, begin: int, end: int) -> np.ndarray:
        if self.fused is not None:
            base = expert * 2 * self.width * self.hidden
            offset = 0 if kind == "gate" else self.width * self.hidden
            return self.fused.read(base + offset + begin, base + offset + end)
        view = self.gate if kind == "gate" else self.up
        base = expert * self.width * self.hidden
        return view.read(base + begin, base + end)

    def read(self, begin: int, end: int) -> np.ndarray:
        out: list[np.ndarray] = []
        span = 2 * self.width * self.hidden
        half_span = self.width * self.hidden
        while begin < end:
            expert, local = divmod(begin, span)
            stop = min(end, (expert + 1) * span)
            if local < half_span:
                take = min(stop - begin, half_span - local)
                out.append(self._half("gate", expert, local, local + take))
            else:
                take = min(stop - begin, span - local)
                out.append(self._half("up", expert, local - half_span,
                                      local - half_span + take))
            begin += take
        return np.concatenate(out) if out else _empty("BF16")


class _Reshape(View):
    """Same flat order, a different logical shape (a size-1 axis added)."""

    def __init__(self, inner: View, shape: tuple[int, ...]) -> None:
        if int(np.prod(shape)) != int(np.prod(inner.shape)):
            raise ValueError("reshape changes the element count")
        self.inner = inner
        self.shape = shape
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        return self.inner.read(begin, end)



# ------------------------------------------------------------------- head-order permutation


def tiled_to_grouped(count: int, key_heads: int, per_key: int, head_dim: int) -> np.ndarray:
    """Inverse of llama.cpp's V-head tiling.

    GGUF (tiled): position = head_in_group * key_heads + key_head.
    HF  (grouped): position = key_head * heads_per_group + head_in_group.
    """
    if key_heads * per_key * head_dim != count:
        raise ValueError(
            f"V-head order: {key_heads} key heads x {per_key} value heads x {head_dim} does not "
            f"cover {count} rows"
        )
    grouped = np.arange(count).reshape(key_heads, per_key, head_dim)
    return np.transpose(grouped, (1, 0, 2)).reshape(-1)
