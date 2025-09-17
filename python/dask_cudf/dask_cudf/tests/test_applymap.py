# Copyright (c) 2022-2025, NVIDIA CORPORATION.

import pytest
from pandas import NA

from dask import dataframe as dd

from cudf.core._compat import PANDAS_GE_210

from dask_cudf.tests.utils import _make_random_frame


@pytest.mark.parametrize(
    "func, meta",
    [
        (lambda x: x + 1, {"x": "int64", "y": "int64"}),
        (lambda x: x - 0.5, {"x": "float64", "y": "float64"}),
        (
            lambda x: 2 if x is NA else 2 + (x + 1) / 4.1,
            {"x": "float64", "y": "float64"},
        ),
        (lambda x: 42, {"x": "int64", "y": "int64"}),
    ],
)
@pytest.mark.parametrize("has_na", [True, False])
@pytest.mark.skipif(
    not PANDAS_GE_210,
    reason="DataFrame.map requires pandas>=2.1.0",
)
def test_applymap_basic(func, has_na, meta):
    size = 2000
    pdf, dgdf = _make_random_frame(size, include_na=False)

    dpdf = dd.from_pandas(pdf, npartitions=dgdf.npartitions)

    expect = dpdf.map(func, meta=meta)
    got = dgdf.map(func, meta=meta)
    dd.assert_eq(expect, got, check_dtype=False)
