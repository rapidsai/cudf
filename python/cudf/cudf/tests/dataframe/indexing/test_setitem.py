# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_listcol_setitem_retain_dtype():
    df = cudf.DataFrame(
        {"a": cudf.Series([["a", "b"], []]), "b": [1, 2], "c": [123, 321]}
    )
    df1 = df.head(0)
    # Performing a setitem on `b` triggers a `column.column_empty` call
    # which tries to create an empty ListColumn.
    df1["b"] = df1["c"]
    # Performing a copy to trigger a copy dtype which is obtained by accessing
    # `ListColumn.children` that would have been corrupted in previous call
    # prior to this fix: https://github.com/rapidsai/cudf/pull/10151/
    df2 = df1.copy()
    assert df2["a"].dtype == df["a"].dtype


def test_setitem_datetime():
    df = cudf.DataFrame({"date": pd.date_range("20010101", "20010105").values})
    assert df.date.dtype.kind == "M"


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)


def test_dataframe_cow_slice_setitem():
    with cudf.option_context("copy_on_write", True):
        df = cudf.DataFrame(
            {"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}
        )
        slice_df = df[1:4]

        assert_eq(
            slice_df,
            cudf.DataFrame(
                {"a": [11, 12, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
            ),
        )

        slice_df["a"][2] = 1111

        assert_eq(
            slice_df,
            cudf.DataFrame(
                {"a": [11, 1111, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
            ),
        )
        assert_eq(
            df,
            cudf.DataFrame(
                {"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}
            ),
        )
