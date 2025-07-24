# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(TypeError):
        iter(gs._column)


@pytest.mark.parametrize("data", [[], [None, None], ["a", None]])
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


def test_set_index_unequal_length():
    s = cudf.Series(dtype="float64")
    with pytest.raises(ValueError):
        s.index = [1, 2, 3]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4],
        ["a", "b", "c"],
        [1.2, 2.2, 4.5],
        [np.nan, np.nan],
        [None, None, None],
    ],
)
def test_axes(data):
    csr = cudf.Series(data)
    psr = csr.to_pandas()

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        pytest.param(
            [np.nan, 10, 15, 16],
            marks=pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/49818"
            ),
        ),
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
        [],
    ],
)
def test_series_hasnans(data):
    gs = cudf.Series(data, nan_as_null=False)
    ps = gs.to_pandas(nullable=True)

    # Check type to avoid mixing Python bool and NumPy bool
    assert isinstance(gs.hasnans, bool)
    assert gs.hasnans == ps.hasnans
