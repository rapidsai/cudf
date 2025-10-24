# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize(
    "column_names",
    [
        ["v0", "v1"],
        ["v0", "index"],
        pd.MultiIndex.from_tuples([("x0", "x1"), ("y0", "y1")]),
        pd.MultiIndex.from_tuples([(1, 2), (10, 11)], names=["ABC", "DEF"]),
    ],
)
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index(level, drop, column_names, inplace, col_level, col_fill):
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    pdf = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]], index=midx, columns=column_names
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_dup_level_name(level, drop, inplace, col_level, col_fill):
    # midx levels are named [None, None]
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    pdf = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=midx)
    gdf = cudf.from_pandas(pdf)
    if level == [None]:
        assert_exceptions_equal(
            lfunc=pdf.reset_index,
            rfunc=gdf.reset_index,
            lfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
        )
        return

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_named(drop, inplace, col_level, col_fill):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf.index.name = "cudf"
    gdf.index.name = "cudf"

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


@pytest.mark.parametrize("column_names", [["x", "y"], ["index", "y"]])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_unnamed(drop, inplace, column_names, col_level, col_fill):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf.columns = column_names
    gdf.columns = column_names

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


def test_reset_index_invalid_level():
    with pytest.raises(IndexError):
        cudf.DataFrame([1]).reset_index(level=2)

    with pytest.raises(IndexError):
        pd.DataFrame([1]).reset_index(level=2)
