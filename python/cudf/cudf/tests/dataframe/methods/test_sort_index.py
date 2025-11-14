# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import itertools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_220,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq


def test_dataframe_empty_sort_index():
    pdf = pd.DataFrame({"x": []})
    gdf = cudf.DataFrame(pdf)

    expect = pdf.sort_index()
    got = gdf.sort_index()

    assert_eq(expect, got, check_index_type=True)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(0, 3, 1),
        [3.0, 1.0, np.nan],
        # Test for single column MultiIndex
        pd.MultiIndex.from_arrays(
            [
                [2, 0, 1],
            ]
        ),
        pd.RangeIndex(2, -1, -1),
    ],
)
@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_sort_index(
    request, index, axis, ascending, inplace, ignore_index, na_position
):
    if not PANDAS_GE_220 and axis in (1, "columns") and ignore_index:
        pytest.skip(reason="Bug fixed in pandas-2.2")

    pdf = pd.DataFrame(
        {"b": [1, 3, 2], "a": [1, 4, 3], "c": [4, 1, 5]},
        index=index,
    )
    gdf = cudf.DataFrame(pdf)

    expected = pdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )
    got = gdf.sort_index(
        axis=axis,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(pdf, gdf, check_index_type=True)
    else:
        assert_eq(expected, got, check_index_type=True)


@pytest.mark.parametrize("axis", [0, 1, "index", "columns"])
@pytest.mark.parametrize(
    "level",
    [
        0,
        "b",
        1,
        ["b"],
        "a",
        ["a", "b"],
        ["b", "a"],
        [0, 1],
        [1, 0],
        [0, 2],
        None,
    ],
)
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_dataframe_mulitindex_sort_index(
    request, axis, level, ascending, inplace, ignore_index, na_position
):
    request.applymarker(
        pytest.mark.xfail(
            condition=axis in (1, "columns")
            and level is None
            and not ascending
            and ignore_index,
            reason="https://github.com/pandas-dev/pandas/issues/57293",
        )
    )
    pdf = pd.DataFrame(
        {
            "b": [1.0, 3.0, np.nan],
            "a": [1, 4, 3],
            1: ["a", "b", "c"],
            "e": [3, 1, 4],
            "d": [1, 2, 8],
        }
    ).set_index(["b", "a", 1])
    gdf = cudf.DataFrame(pdf)

    expected = pdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        inplace=inplace,
        na_position=na_position,
        ignore_index=ignore_index,
    )
    got = gdf.sort_index(
        axis=axis,
        level=level,
        ascending=ascending,
        ignore_index=ignore_index,
        inplace=inplace,
        na_position=na_position,
    )

    if inplace is True:
        assert_eq(pdf, gdf)
    else:
        assert_eq(expected, got)


def test_sort_index_axis_1_ignore_index_true_columnaccessor_state_names():
    gdf = cudf.DataFrame([[1, 2, 3]], columns=["b", "a", "c"])
    result = gdf.sort_index(axis=1, ignore_index=True)
    assert result._data.names == tuple(result._data.keys())


@pytest.mark.parametrize(
    "levels",
    itertools.chain.from_iterable(
        itertools.permutations(range(3), n) for n in range(1, 4)
    ),
    ids=str,
)
def test_multiindex_sort_index_partial(levels):
    df = pd.DataFrame(
        {
            "a": [3, 3, 3, 1, 1, 1, 2, 2],
            "b": [4, 2, 7, -1, 11, -2, 7, 7],
            "c": [4, 4, 2, 3, 3, 3, 1, 1],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    ).set_index(["a", "b", "c"])
    cdf = cudf.from_pandas(df)

    expect = df.sort_index(level=levels, sort_remaining=True)
    got = cdf.sort_index(level=levels, sort_remaining=True)
    assert_eq(expect, got)


def test_df_cat_sort_index():
    df = cudf.DataFrame(
        {
            "a": pd.Categorical(list("aababcabbc"), categories=list("abc")),
            "b": np.arange(10),
        }
    )

    got = df.set_index("a").sort_index()
    expect = df.to_pandas().set_index("a").sort_index()

    assert_eq(got, expect)
