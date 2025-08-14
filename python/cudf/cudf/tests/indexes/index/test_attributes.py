# Copyright (c) 2025, NVIDIA CORPORATION.

import re

import numpy as np
import pandas as pd
import pytest

import cudf


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 4, None],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        [1, 2, 3, 4, np.nan],
        [10, 9, 8, np.nan, 7],
        [10, 9, 8, 8, 7, np.nan],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["c", "d", "e", "f", None],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_index_is_unique_monotonic(testlist):
    index = cudf.Index(testlist)
    index_pd = pd.Index(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing


def test_name():
    idx = cudf.Index(np.asarray([4, 5, 6, 10]), name="foo")
    assert idx.name == "foo"


def test_index_names():
    idx = cudf.Index([1, 2, 3], name="idx")
    assert idx.names == ("idx",)


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
def test_index_empty(data, all_supported_types_as_str):
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    assert pdi.empty == gdi.empty


@pytest.mark.parametrize("data", [[1, 2, 3, 4], []])
def test_index_size(data, all_supported_types_as_str):
    pdi = pd.Index(data, dtype=all_supported_types_as_str)
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    assert pdi.size == gdi.size


@pytest.mark.parametrize("data", [[], [1]])
def test_index_iter_error(data, all_supported_types_as_str):
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gdi.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gdi)


@pytest.mark.parametrize("data", [[], [1]])
def test_index_values_host(data, all_supported_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            len(data) > 0
            and all_supported_types_as_str
            in {"timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"},
            reason=f"wrong result for {all_supported_types_as_str}",
        )
    )
    gdi = cudf.Index(data, dtype=all_supported_types_as_str)
    pdi = pd.Index(data, dtype=all_supported_types_as_str)

    np.testing.assert_array_equal(gdi.values_host, pdi.values)
