# Copyright (c) 2025, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize(
    "args,gd_kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex(copy, args, gd_kwargs):
    reindex_data = cudf.datasets.randomdata(
        nrows=6,
        dtypes={
            "a": "category",
            "c": float,
            "d": str,
        },
    )
    pdf, gdf = reindex_data.to_pandas(), reindex_data
    assert_eq(
        pdf.reindex(*args, **gd_kwargs, copy=True),
        gdf.reindex(*args, **gd_kwargs, copy=copy),
    )


@pytest.mark.parametrize("fill_value", [-1.0, 0.0, 1.5])
@pytest.mark.parametrize(
    "args,kwargs",
    [
        ([], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {}),
        ([[-3, 0, 3, 0, -2, 1, 3, 4, 6]], {"axis": 0}),
        ([["a", "b", "c", "d", "e"]], {"axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": 0}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": 1}),
        ([], {"labels": [-3, 0, 3, 0, -2, 1, 3, 4, 6], "axis": "index"}),
        ([], {"labels": ["a", "b", "c", "d", "e"], "axis": "columns"}),
        ([], {"index": [-3, 0, 3, 0, -2, 1, 3, 4, 6]}),
        ([], {"columns": ["a", "b", "c", "d", "e"]}),
        (
            [],
            {
                "index": [-3, 0, 3, 0, -2, 1, 3, 4, 6],
                "columns": ["a", "b", "c", "d", "e"],
            },
        ),
    ],
)
def test_dataframe_reindex_fill_value(args, kwargs, fill_value):
    reindex_data_numeric = cudf.datasets.randomdata(
        nrows=6,
        dtypes={"a": float, "b": float, "c": float},
    )
    pdf, gdf = reindex_data_numeric.to_pandas(), reindex_data_numeric
    assert_eq(
        pdf.reindex(*args, **kwargs, fill_value=fill_value),
        gdf.reindex(*args, **kwargs, fill_value=fill_value),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_dataframe_reindex_change_dtype(copy):
    index = pd.date_range("12/29/2009", periods=10, freq="D")
    columns = ["a", "b", "c", "d", "e"]
    gdf = cudf.datasets.randomdata(
        nrows=6, dtypes={"a": "category", "c": float, "d": str}
    )
    pdf = gdf.to_pandas()
    # Validate reindexes both labels and column names when
    # index=index_labels and columns=column_labels
    assert_eq(
        pdf.reindex(index=index, columns=columns, copy=True),
        gdf.reindex(index=index, columns=columns, copy=copy),
        check_freq=False,
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_categorical_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"a": "category"})
    pdf = gdf.to_pandas()
    assert_eq(pdf["a"].reindex(copy=True), gdf["a"].reindex(copy=copy))
    assert_eq(
        pdf["a"].reindex(index, copy=True), gdf["a"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["a"].reindex(index=index, copy=True),
        gdf["a"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_float_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"c": float})
    pdf = gdf.to_pandas()
    assert_eq(pdf["c"].reindex(copy=True), gdf["c"].reindex(copy=copy))
    assert_eq(
        pdf["c"].reindex(index, copy=True), gdf["c"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["c"].reindex(index=index, copy=True),
        gdf["c"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("copy", [True, False])
def test_series_string_reindex(copy):
    index = [-3, 0, 3, 0, -2, 1, 3, 4, 6]
    gdf = cudf.datasets.randomdata(nrows=6, dtypes={"d": str})
    pdf = gdf.to_pandas()
    assert_eq(pdf["d"].reindex(copy=True), gdf["d"].reindex(copy=copy))
    assert_eq(
        pdf["d"].reindex(index, copy=True), gdf["d"].reindex(index, copy=copy)
    )
    assert_eq(
        pdf["d"].reindex(index=index, copy=True),
        gdf["d"].reindex(index=index, copy=copy),
    )


@pytest.mark.parametrize("names", [None, ["a", "b"]])
@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_reindex_multiindex_col_to_multiindex(names, klass):
    idx = pd.Index(
        [("A", "one"), ("A", "two")],
        dtype="object",
    )
    df = pd.DataFrame([[1, 2]], columns=idx)
    gdf = cudf.from_pandas(df)
    midx = klass.from_tuples([("A", "one"), ("A", "three")], names=names)
    result = gdf.reindex(columns=midx)
    expected = cudf.DataFrame([[1, None]], columns=midx)
    # (pandas2.0): check_dtype=False won't be needed
    # as None col will return object instead of float
    assert_eq(result, expected, check_dtype=False)


@pytest.mark.parametrize("names", [None, ["a", "b"]])
@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_reindex_tuple_col_to_multiindex(names, klass):
    idx = pd.Index(
        [("A", "one"), ("A", "two")], dtype="object", tupleize_cols=False
    )
    df = pd.DataFrame([[1, 2]], columns=idx)
    gdf = cudf.from_pandas(df)
    midx = klass.from_tuples([("A", "one"), ("A", "two")], names=names)
    result = gdf.reindex(columns=midx)
    expected = cudf.DataFrame([[1, 2]], columns=midx)
    assert_eq(result, expected)


@pytest.mark.parametrize("name", [None, "foo"])
@pytest.mark.parametrize("klass", [range, cudf.RangeIndex, pd.RangeIndex])
def test_reindex_columns_rangeindex_keeps_rangeindex(name, klass):
    new_columns = klass(3)
    exp_name = None
    if klass is not range:
        new_columns.name = name
        exp_name = name
    df = cudf.DataFrame([[1, 2]])
    result = df.reindex(columns=new_columns).columns
    expected = pd.RangeIndex(3, name=exp_name)
    assert_eq(result, expected)
