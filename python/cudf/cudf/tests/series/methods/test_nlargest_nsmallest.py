# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.parametrize("attr", ["nlargest", "nsmallest"])
def test_series_nlargest_nsmallest_str_error(attr):
    gs = cudf.Series(["a", "b", "c", "d", "e"])
    ps = gs.to_pandas()

    assert_exceptions_equal(
        getattr(gs, attr), getattr(ps, attr), ([], {"n": 1}), ([], {"n": 1})
    )


@pytest.mark.parametrize("data", [[0, 1, 1, 2, 2, 2, 3, 3], [0], [1, 2, 3]])
@pytest.mark.parametrize("n", [-100, -2, 0, 1, 4])
def test_series_nlargest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = cudf.Series(data)
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
@pytest.mark.parametrize("n", [-100, -2, 0, 1, 4])
def test_series_nsmallest(data, n):
    """Indirectly tests Series.sort_values()"""
    sr = cudf.Series(data)
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
