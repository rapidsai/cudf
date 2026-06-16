# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level tests for the MLIR backend MaskedType *lowering* (PR 2 MVP).

Each test JIT-compiles a small ``cuda.jit`` kernel that constructs a
``Masked`` value, accesses its ``.value`` and ``.valid`` fields, and
writes the results to host arrays for comparison. Together they cover:

* the data model registration (struct ``{value_ty, i1}``);
* ``Masked(value, valid)`` constructor lowering;
* the generic ``.value`` / ``.valid`` getattr lowering.

Out of scope (later PRs): ``pack_return``, NA, masked binary/unary/
comparison ops, masked-to-masked unification, datetime / timedelta /
string value types.
"""

from __future__ import annotations

import warnings

import cupy as cp
import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import (
    cuda,
    types,
)

import cudf.core.udf.mlir_backend.masked_lowering

# Importing these registers typing/lowering on numba_cuda_mlir's registries.
import cudf.core.udf.mlir_backend.masked_typing  # noqa: F401
from cudf.core.udf.api import Masked
from cudf.core.udf.utils import DEPRECATED_SM_REGEX
from cudf.utils._numba import _CUDFNumbaConfig


def _jit(sig):
    """Compile a small kernel suppressing the deprecated-SM warning."""

    def deco(fn):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=DEPRECATED_SM_REGEX,
                category=UserWarning,
                module=r"^numba\.cuda(\.|$)",
            )
            return cuda.jit(sig)(fn)

    return deco


def _launch(kernel, *args):
    """Launch ``kernel[1, 1](*args)`` with the low-occupancy warning suppressed."""
    with _CUDFNumbaConfig():
        kernel[1, 1](*args)
    cuda.synchronize()


# Combinations of (numba type, numpy dtype, sample value) used to
# parametrize the multi-dtype tests below.
_DTYPE_SAMPLES = [
    (types.int8, np.int8, np.int8(7)),
    (types.int16, np.int16, np.int16(-300)),
    (types.int32, np.int32, np.int32(123_456)),
    (types.int64, np.int64, np.int64(-9_999_999)),
    (types.uint8, np.uint8, np.uint8(255)),
    (types.uint32, np.uint32, np.uint32(4_000_000_000)),
    (types.uint64, np.uint64, np.uint64(9_223_372_036_854_775_808)),
    (types.float32, np.float32, np.float32(3.5)),
    (types.float64, np.float64, np.float64(-1.25)),
    (types.boolean, np.bool_, np.bool_(True)),
]


@pytest.mark.parametrize("nb_ty,np_dt,sample", _DTYPE_SAMPLES)
def test_masked_constructor_and_accessors_valid_true(nb_ty, np_dt, sample):
    """``Masked(v, True).value == v`` and ``.valid == True``, for every supported dtype."""

    @_jit(types.void(nb_ty[::1], types.boolean[::1], nb_ty[::1]))
    def k(out_value, out_valid, v):
        m = Masked(v[0], True)
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_v = cp.array([sample], dtype=np_dt)
    out_value = cp.zeros(1, dtype=np_dt)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v)
    assert out_value.get()[0] == sample
    assert bool(out_valid.get()[0]) is True


@pytest.mark.parametrize("nb_ty,np_dt,sample", _DTYPE_SAMPLES)
def test_masked_constructor_and_accessors_valid_false(nb_ty, np_dt, sample):
    """``Masked(v, False).valid == False`` (and ``.value`` is preserved)."""

    @_jit(types.void(nb_ty[::1], types.boolean[::1], nb_ty[::1]))
    def k(out_value, out_valid, v):
        m = Masked(v[0], False)
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_v = cp.array([sample], dtype=np_dt)
    out_value = cp.zeros(1, dtype=np_dt)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v)
    assert out_value.get()[0] == sample
    assert bool(out_valid.get()[0]) is False


def test_masked_constructor_with_runtime_valid_flag():
    """The validity flag can be a runtime value, not only a Python literal."""

    @_jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.int64[::1],
            types.boolean[::1],
        )
    )
    def k(out_value, out_valid, v, valid):
        m = Masked(v[0], valid[0])
        out_value[0] = m.value
        out_valid[0] = m.valid

    in_v = cp.array([42], dtype=np.int64)
    in_valid = cp.array([False], dtype=np.bool_)
    out_value = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)
    _launch(k, out_value, out_valid, in_v, in_valid)
    assert int(out_value.get()[0]) == 42
    assert bool(out_valid.get()[0]) is False
