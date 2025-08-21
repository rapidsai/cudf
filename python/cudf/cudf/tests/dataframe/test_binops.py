# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "expected",
    [
        pd.RangeIndex(1, 2, name="a"),
        pd.Index([1], dtype=np.int8, name="a"),
        pd.MultiIndex.from_arrays([[1]], names=["a"]),
    ],
)
@pytest.mark.parametrize("binop", [lambda df: df == df, lambda df: df - 1])
def test_dataframe_binop_preserves_column_metadata(expected, binop):
    df = cudf.DataFrame([1], columns=expected)
    result = binop(df).columns
    pd.testing.assert_index_equal(result, expected, exact=True)


def test_dataframe_series_dot():
    pser = pd.Series(range(2))
    gser = cudf.from_pandas(pser)

    expected = pser @ pser
    actual = gser @ gser

    assert_eq(expected, actual)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"))
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pser], {}),
        rfunc_args_and_kwargs=([gser], {}),
    )

    assert_exceptions_equal(
        lfunc=pdf.dot,
        rfunc=gdf.dot,
        lfunc_args_and_kwargs=([pdf], {}),
        rfunc_args_and_kwargs=([gdf], {}),
    )

    pser = pd.Series(range(2), index=["a", "k"])
    gser = cudf.from_pandas(pser)

    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("ab"), index=["a", "k"])
    gdf = cudf.from_pandas(pdf)

    expected = pser @ pdf
    actual = gser @ gdf

    assert_eq(expected, actual)

    actual = gdf @ [2, 3]
    expected = pdf @ [2, 3]

    assert_eq(expected, actual)

    actual = pser @ [12, 13]
    expected = gser @ [12, 13]

    assert_eq(expected, actual)


def test_dataframe_binop_with_datetime_index():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(
        rng.random(2),
        index=pd.Index(
            [
                "2000-01-04",
                "2000-01-03",
            ],
            dtype="datetime64[ns]",
        ),
    )
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


def test_dataframe_binop_and_where():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(rng.random(size=(2, 2)), columns=pd.Index([True, False]))
    gdf = cudf.from_pandas(df)

    expected = df > 1
    got = gdf > 1

    assert_eq(expected, got)

    expected = df[df > 1]
    got = gdf[gdf > 1]

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_string_types():
    rng = np.random.default_rng(seed=0)
    df1 = pd.DataFrame(rng.random(size=(3, 3)), columns=pd.Index([0, 1, 2]))
    df2 = pd.DataFrame(
        rng.random(size=(6, 6)),
        columns=pd.Index([0, 1, 2, "VhDoHxRaqt", "X0NNHBIPfA", "5FbhPtS0D1"]),
    )
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    expected = df2 + df1
    got = gdf2 + gdf1

    assert_eq(expected, got)


def test_dataframe_binop_with_mixed_date_types():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        rng.random(size=(2, 2)),
        columns=pd.Index(["2000-01-03", "2000-01-04"], dtype="datetime64[ns]"),
    )
    ser = pd.Series(rng.random(size=3), index=[0, 1, 2])
    gdf = cudf.from_pandas(df)
    gser = cudf.from_pandas(ser)
    expected = df - ser
    got = gdf - gser
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "df1",
    [
        pd.DataFrame({"a": [10, 11, 12]}, index=["a", "b", "z"]),
        pd.DataFrame({"z": ["a"]}),
        pd.DataFrame({"a": [], "b": []}),
    ],
)
@pytest.mark.parametrize(
    "df2",
    [
        pd.DataFrame(),
        pd.DataFrame({"a": ["a", "a", "c", "z", "A"], "z": [1, 2, 3, 4, 5]}),
    ],
)
def test_dataframe_error_equality(df1, df2, comparison_op):
    gdf1 = cudf.from_pandas(df1)
    gdf2 = cudf.from_pandas(df2)

    assert_exceptions_equal(
        comparison_op, comparison_op, ([df1, df2],), ([gdf1, gdf2],)
    )
