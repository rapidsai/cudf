# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd

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


def test_setitem_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result["a"] = [2]
    expected["a"] = [2]
    assert_eq(result, expected)


def test_dataframe_assign_scalar_to_empty_series():
    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame({"a": []})
    expected.a = 0
    actual.a = 0
    assert_eq(expected, actual)


def test_dataframe_assign_cp_np_array():
    m, n = 5, 3
    cp_ndarray = cp.random.randn(m, n)
    pdf = pd.DataFrame({f"f_{i}": range(m) for i in range(n)})
    gdf = cudf.DataFrame({f"f_{i}": range(m) for i in range(n)})
    pdf[[f"f_{i}" for i in range(n)]] = cp.asnumpy(cp_ndarray)
    gdf[[f"f_{i}" for i in range(n)]] = cp_ndarray

    assert_eq(pdf, gdf)


def test_dataframe_setitem_cupy_array():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.standard_normal(size=(10, 2)))
    gdf = cudf.from_pandas(pdf)

    gpu_array = cp.array([True, False] * 5)
    pdf[gpu_array.get()] = 1.5
    gdf[gpu_array] = 1.5

    assert_eq(pdf, gdf)
