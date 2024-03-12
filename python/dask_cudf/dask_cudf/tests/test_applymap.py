# Copyright (c) 2022-2024, NVIDIA CORPORATION.

import pytest
from pandas import NA

from dask import dataframe as dd

from dask_cudf.tests.utils import _make_random_frame


@pytest.mark.parametrize(
    "func",
    [
        lambda x: x + 1,
        lambda x: x - 0.5,
        lambda x: 2 if x is NA else 2 + (x + 1) / 4.1,
        lambda x: 42,
    ],
)
@pytest.mark.parametrize("has_na", [True, False])
def test_applymap_basic(func, has_na):
    size = 2000
    pdf, dgdf = _make_random_frame(size, include_na=False)

    dpdf = dd.from_pandas(pdf, npartitions=dgdf.npartitions)

    expect = dpdf.map(func)
    got = dgdf.map(func)
    dd.assert_eq(expect, got, check_dtype=False)
