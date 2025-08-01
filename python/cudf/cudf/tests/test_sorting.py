# Copyright (c) 2018-2025, NVIDIA CORPORATION.


import numpy as np
import pandas as pd
import pytest

from cudf import Series
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.fixture(params=[2, 257])
def nelem(request):
    return request.param


@pytest.fixture(
    params=[
        np.int32,
        np.int64,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    ]
)
def dtype(request):
    return request.param


@pytest.fixture(params=[slice(1, None), slice(None, -1), slice(1, -1)])
def sliceobj(request):
    return request.param


@pytest.mark.parametrize("ignore_index", [True, False])
def test_series_sort_values_ignore_index(ignore_index):
    gsr = Series([1, 3, 5, 2, 4])
    psr = gsr.to_pandas()

    expect = psr.sort_values(ignore_index=ignore_index)
    got = gsr.sort_values(ignore_index=ignore_index)
    assert_eq(expect, got)


@pytest.mark.parametrize("asc", [True, False])
def test_series_argsort(nelem, dtype, asc):
    rng = np.random.default_rng(seed=0)
    sr = Series((100 * rng.random(nelem)).astype(dtype))
    res = sr.argsort(ascending=asc)

    if asc:
        expected = np.argsort(sr.to_numpy(), kind="mergesort")
    else:
        # -1 multiply works around missing desc sort (may promote to float64)
        expected = np.argsort(sr.to_numpy() * np.int8(-1), kind="mergesort")
    np.testing.assert_array_equal(expected, res.to_numpy())


@pytest.mark.parametrize("asc", [True, False])
def test_series_sort_index(nelem, asc):
    rng = np.random.default_rng(seed=0)
    sr = Series(100 * rng.random(nelem))
    psr = sr.to_pandas()

    expected = psr.sort_index(ascending=asc)
    got = sr.sort_index(ascending=asc)

    assert_eq(expected, got)


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -50, -12, -2, 0, 1, 2, 3, 4, 7])
def test_series_nlargest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nlargest(n), psr.nlargest(n))
    assert_eq(sr.nlargest(n, keep="last"), psr.nlargest(n, keep="last"))

    assert_exceptions_equal(
        lfunc=psr.nlargest,
        rfunc=sr.nlargest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -50, -12, -2, 0, 1, 2, 3, 4, 9])
def test_series_nsmallest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = Series(data)
    psr = pd.Series(data)
    assert_eq(sr.nsmallest(n), psr.nsmallest(n))
    assert_eq(
        sr.nsmallest(n, keep="last").sort_index(),
        psr.nsmallest(n, keep="last").sort_index(),
    )

    assert_exceptions_equal(
        lfunc=psr.nsmallest,
        rfunc=sr.nsmallest,
        lfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
        rfunc_args_and_kwargs=([], {"n": 3, "keep": "what"}),
    )
