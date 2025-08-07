# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
    expect_warning_if,
)


@pytest.mark.parametrize(
    "pframe, replace_args",
    [
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([5, 1, 2, 3, 4]),
            {"to_replace": {5: 0, 3: -5}, "inplace": True},
        ),
        (pd.Series([5, 1, 2, 3, 4]), {}),
        pytest.param(
            pd.Series(["one", "two", "three"], dtype="category"),
            {"to_replace": "one", "value": "two", "inplace": True},
            marks=pytest.mark.xfail(
                condition=PANDAS_VERSION >= PANDAS_CURRENT_SUPPORTED_VERSION,
                reason="https://github.com/pandas-dev/pandas/issues/43232"
                "https://github.com/pandas-dev/pandas/issues/53358",
            ),
        ),
        (
            pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9]}),
            {"to_replace": 5, "value": 0, "inplace": True},
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": True,
            },
        ),
        (
            pd.Series([1, 2, 3, 45]),
            {
                "to_replace": np.array([]).astype(int),
                "value": 77,
                "inplace": False,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": {"a": 2}, "value": {"a": -33}, "inplace": True},
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {
                "to_replace": {"a": [2, 5]},
                "value": {"a": [9, 10]},
                "inplace": True,
            },
        ),
        (
            pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]}),
            {"to_replace": [], "value": [], "inplace": True},
        ),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Warning not given on older versions of pandas",
)
def test_replace_inplace(pframe, replace_args):
    gpu_frame = cudf.from_pandas(pframe)
    pandas_frame = pframe.copy()

    gpu_copy = gpu_frame.copy()
    cpu_copy = pandas_frame.copy()

    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)
    with expect_warning_if(len(replace_args) == 0):
        gpu_frame.replace(**replace_args)
    with expect_warning_if(len(replace_args) == 0):
        pandas_frame.replace(**replace_args)
    assert_eq(gpu_frame, pandas_frame)
    assert_eq(gpu_copy, cpu_copy)


def test_replace_df_error():
    pdf = pd.DataFrame({"a": [1, 2, 3, 4, 5, 666]})
    gdf = cudf.from_pandas(pdf)

    assert_exceptions_equal(
        lfunc=pdf.replace,
        rfunc=gdf.replace,
        lfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
        rfunc_args_and_kwargs=([], {"to_replace": -1, "value": []}),
    )


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        ([2, 7.4], [4, 7.9]),
        ([2, 7.4], None),
        (
            None,
            [4, 7.9],
        ),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_clip(lower, upper, inplace):
    pdf = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )
    gdf = cudf.from_pandas(pdf)

    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)
    expect = pdf.clip(lower=lower, upper=upper, axis=1)

    if inplace is True:
        assert_eq(expect, gdf)
    else:
        assert_eq(expect, got)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [("b", "d"), ("b", None), (None, "c"), (None, None)],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_dataframe_category_clip(lower, upper, inplace):
    data = ["a", "b", "c", "d", "e"]
    pdf = pd.DataFrame({"a": data})
    gdf = cudf.from_pandas(pdf)
    gdf["a"] = gdf["a"].astype("category")

    expect = pdf.clip(lower=lower, upper=upper)
    got = gdf.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gdf.astype("str"))
    else:
        assert_eq(expect, got.astype("str"))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [([2, 7.4], [4, 7.9, "d"]), ([2, 7.4, "a"], [4, 7.9, "d"])],
)
def test_dataframe_exceptions_for_clip(lower, upper):
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [7.1, 7.24, 7.5, 7.8, 8.11]}
    )

    with pytest.raises(ValueError):
        gdf.clip(lower=lower, upper=upper)


@pytest.mark.parametrize(
    ("data", "lower", "upper"),
    [
        ([1, 2, 3, 4, 5], 2, 4),
        ([1, 2, 3, 4, 5], 2, None),
        ([1, 2, 3, 4, 5], None, 4),
        ([1, 2, 3, 4, 5], None, None),
        ([1, 2, 3, 4, 5], 4, 2),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 4, 2),
        (pd.Series([1, 2, 3, 4, 5], dtype="int32"), 4, 2),
        (["a", "b", "c", "d", "e"], "b", "d"),
        (["a", "b", "c", "d", "e"], "b", None),
        (["a", "b", "c", "d", "e"], None, "d"),
        (["a", "b", "c", "d", "e"], "d", "b"),
    ],
)
@pytest.mark.parametrize("inplace", [True, False])
def test_series_clip(data, lower, upper, inplace):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    expect = psr.clip(lower=lower, upper=upper)
    got = gsr.clip(lower=lower, upper=upper, inplace=inplace)

    if inplace is True:
        assert_eq(expect, gsr)
    else:
        assert_eq(expect, got)


def test_series_exceptions_for_clip():
    with pytest.raises(ValueError):
        cudf.Series([1, 2, 3, 4]).clip([1, 2], [2, 3])

    with pytest.raises(NotImplementedError):
        cudf.Series([1, 2, 3, 4]).clip(1, 2, axis=0)


@pytest.mark.parametrize(
    "data", [[1, 2.0, 3, 4, None, 1, None, 10, None], ["a", "b", "c"]]
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        [1, 2, 3],
        ["a", "b", "z"],
        ["a", "b", "c", "d", "e", "f", "g", "l", "m"],
    ],
)
@pytest.mark.parametrize("value", [[1, 2, 3, 4, None, 1, None, 10, None]])
def test_series_fillna(data, index, value):
    psr = pd.Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )
    gsr = cudf.Series(
        data,
        index=index if index is not None and len(index) == len(data) else None,
    )

    expect = psr.fillna(pd.Series(value))
    got = gsr.fillna(cudf.Series(value))
    assert_eq(expect, got)


def test_series_fillna_error():
    psr = pd.Series([1, 2, None, 3, None])
    gsr = cudf.from_pandas(psr)

    assert_exceptions_equal(
        psr.fillna,
        gsr.fillna,
        ([pd.DataFrame({"a": [1, 2, 3]})],),
        ([cudf.DataFrame({"a": [1, 2, 3]})],),
    )


def test_series_replace_errors():
    gsr = cudf.Series([1, 2, None, 3, None])
    psr = gsr.to_pandas()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: object"
        ),
    ):
        gsr.replace(1, "a")

    gsr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "to_replace and value should be of same types,"
            "got to_replace dtype: int64 and "
            "value dtype: object"
        ),
    ):
        gsr.replace([1, 2], ["a", "b"])

    assert_exceptions_equal(
        psr.replace,
        gsr.replace,
        ([{"a": 1}, 1],),
        ([{"a": 1}, 1],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([[1, 2], [1]],),
        rfunc_args_and_kwargs=([[1, 2], [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([object(), [1]],),
        rfunc_args_and_kwargs=([object(), [1]],),
    )

    assert_exceptions_equal(
        lfunc=psr.replace,
        rfunc=gsr.replace,
        lfunc_args_and_kwargs=([{"a": 1}, object()],),
        rfunc_args_and_kwargs=([{"a": 1}, object()],),
    )


@pytest.mark.parametrize(
    "gsr,old,new,expected",
    [
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            None,
            "a",
            lambda: cudf.Series(["a", "b", "c", "a"]),
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [None, "a", "a"],
            ["c", "b", "d"],
            lambda: cudf.Series(["d", "b", "c", "c"]),
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [None, "a"],
            ["b", None],
            lambda: cudf.Series([None, "b", "c", "b"]),
        ),
        (
            lambda: cudf.Series(["a", "b", "c", None]),
            [None, None],
            [None, None],
            lambda: cudf.Series(["a", "b", "c", None]),
        ),
        (
            lambda: cudf.Series([1, 2, None, 3]),
            None,
            10,
            lambda: cudf.Series([1, 2, 10, 3]),
        ),
        (
            lambda: cudf.Series([1, 2, None, 3]),
            [None, 1, 1],
            [3, 2, 4],
            lambda: cudf.Series([4, 2, 3, 3]),
        ),
        (
            lambda: cudf.Series([1, 2, None, 3]),
            [None, 1],
            [2, None],
            lambda: cudf.Series([None, 2, 2, 3]),
        ),
        (
            lambda: cudf.Series(["a", "q", "t", None], dtype="category"),
            None,
            "z",
            lambda: cudf.Series(["a", "q", "t", "z"], dtype="category"),
        ),
        (
            lambda: cudf.Series(["a", "q", "t", None], dtype="category"),
            [None, "a", "q"],
            ["z", None, None],
            lambda: cudf.Series([None, None, "t", "z"], dtype="category"),
        ),
        (
            lambda: cudf.Series(["a", None, "t", None], dtype="category"),
            [None, "t"],
            ["p", None],
            lambda: cudf.Series(["a", "p", None, "p"], dtype="category"),
        ),
    ],
)
def test_replace_nulls(gsr, old, new, expected):
    gsr = gsr()
    with expect_warning_if(isinstance(gsr.dtype, cudf.CategoricalDtype)):
        actual = gsr.replace(old, new)
    assert_eq(
        expected().sort_values().reset_index(drop=True),
        actual.sort_values().reset_index(drop=True),
    )


def test_fillna_columns_multiindex():
    columns = pd.MultiIndex.from_tuples([("a", "b"), ("d", "e")])
    pdf = pd.DataFrame(
        {"0": [1, 2, None, 3, None], "1": [None, None, None, None, 4]}
    )
    pdf.columns = columns
    gdf = cudf.from_pandas(pdf)

    expected = pdf.fillna(10)
    actual = gdf.fillna(10)

    assert_eq(expected, actual)


def test_fillna_nan_and_null():
    ser = cudf.Series(pa.array([float("nan"), None, 1.1]), nan_as_null=False)
    result = ser.fillna(2.2)
    expected = cudf.Series([2.2, 2.2, 1.1])
    assert_eq(result, expected)


def test_replace_with_index_objects():
    result = cudf.Series([1, 2]).replace(cudf.Index([1]), cudf.Index([2]))
    expected = pd.Series([1, 2]).replace(pd.Index([1]), pd.Index([2]))
    assert_eq(result, expected)


# Example test function for datetime series replace
def test_replace_datetime_series():
    # Create a pandas datetime series
    pd_series = pd.Series(pd.date_range("20210101", periods=5))
    # Replace a specific datetime value
    pd_result = pd_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    # Create a cudf datetime series
    cudf_series = cudf.Series(pd.date_range("20210101", periods=5))
    # Replace a specific datetime value
    cudf_result = cudf_series.replace(
        pd.Timestamp("2021-01-02"), pd.Timestamp("2021-01-10")
    )

    assert_eq(pd_result, cudf_result)


# Example test function for timedelta series replace
def test_replace_timedelta_series():
    # Create a pandas timedelta series
    pd_series = pd.Series(pd.timedelta_range("1 days", periods=5))
    # Replace a specific timedelta value
    pd_result = pd_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    # Create a cudf timedelta series
    cudf_series = cudf.Series(pd.timedelta_range("1 days", periods=5))
    # Replace a specific timedelta value
    cudf_result = cudf_series.replace(
        pd.Timedelta("2 days"), pd.Timedelta("10 days")
    )

    assert_eq(pd_result, cudf_result)


def test_replace_multiple_rows(datadir):
    path = datadir / "parquet" / "replace_multiple_rows.parquet"
    pdf = pd.read_parquet(path)
    gdf = cudf.read_parquet(path)

    pdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    gdf.replace([np.inf, -np.inf], np.nan, inplace=True)

    assert_eq(pdf, gdf, check_dtype=False)
