# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf


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
