# Copyright (c) 2025, NVIDIA CORPORATION.

import cudf
from cudf.testing import assert_eq


def test_struct_of_struct_loc():
    df = cudf.DataFrame({"col": [{"a": {"b": 1}}]})
    expect = cudf.Series([{"a": {"b": 1}}], name="col")
    assert_eq(expect, df["col"])


def test_dataframe_midx_cols_getitem():
    df = cudf.DataFrame(
        {
            "a": ["a", "b", "c"],
            "b": ["b", "", ""],
            "c": [10, 11, 12],
        }
    )
    df.columns = df.set_index(["a", "b"]).index
    pdf = df.to_pandas()

    expected = df["c"]
    actual = pdf["c"]
    assert_eq(expected, actual)
    df = cudf.DataFrame(
        [[1, 0], [0, 1]],
        columns=[
            ["foo", "foo"],
            ["location", "location"],
            ["x", "y"],
        ],
    )
    df = df.assign(bools=cudf.Series([True, False], dtype="bool"))
    assert_eq(df["bools"], df.to_pandas()["bools"])


def test_multicolumn_item():
    gdf = cudf.DataFrame({"x": range(10), "y": range(10), "z": range(10)})
    gdg = gdf.groupby(["x", "y"]).min()
    gdgT = gdg.T
    pdgT = gdgT.to_pandas()
    assert_eq(gdgT[(0, 0)], pdgT[(0, 0)])
