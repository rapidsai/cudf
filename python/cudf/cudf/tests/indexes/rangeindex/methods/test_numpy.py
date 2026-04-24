# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


def test_rangeindex_to_numpy_args():
    gidx = cudf.RangeIndex(start=1, stop=10, step=2)
    pidx = gidx.to_pandas()

    assert_eq(
        gidx.to_numpy(dtype=np.float64, copy=False, na_value=None),
        pidx.to_numpy(dtype=np.float64, copy=False, na_value=None),
    )


def test_rangeindex_to_numpy_caches_host_array():
    gidx = cudf.RangeIndex(start=0, stop=10, step=1)

    first = gidx.to_numpy(copy=False)
    second = gidx.to_numpy(copy=False)

    assert first is second


def test_rangeindex_to_numpy_copy_true_returns_new_array():
    gidx = cudf.RangeIndex(start=0, stop=10, step=1)

    base = gidx.to_numpy(copy=False)
    copied = gidx.to_numpy(copy=True)

    assert copied is not base
    np.testing.assert_array_equal(copied, base)


def test_rangeindex_to_numpy_copy_on_write():
    gidx = cudf.RangeIndex(start=0, stop=10, step=1)

    with cudf.option_context("copy_on_write", True):
        result = gidx.to_numpy()

    base = gidx.to_numpy(copy=False)
    assert result is not base
    np.testing.assert_array_equal(result, base)


def test_rangeindex_values_host_shares_to_numpy_cache():
    gidx = cudf.RangeIndex(start=0, stop=10, step=1)

    with pytest.warns(FutureWarning):
        host_values = gidx.values_host

    assert gidx.to_numpy(copy=False) is host_values
