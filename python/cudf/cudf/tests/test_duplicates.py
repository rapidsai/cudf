# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import itertools
import random

import numpy as np
import pytest
from pandas import DataFrame, MultiIndex, Series, date_range

import cudf
from cudf import concat
from cudf.testing._utils import (
    _create_pandas_series,
    assert_eq,
    assert_exceptions_equal,
)

# TODO: PANDAS 1.0 support
# Revisit drop_duplicates() tests to update parameters like ignore_index.


def assert_df(g, p):
    # assert_eq() with sorted index of dataframes
    g = g.sort_index()
    p = p.sort_index()
    return assert_eq(g, p)


def assert_df2(g, p):
    assert g.index.dtype == p.index.dtype
    np.testing.assert_equal(g.index.to_numpy(), p.index)
    assert tuple(g.columns) == tuple(p.columns)
    for k in g.columns:
        assert g[k].dtype == p[k].dtype
        np.testing.assert_equal(g[k].to_numpy(), p[k])


# most tests are similar to pandas drop_duplicates


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_duplicated_with_misspelled_column_name(subset):
    df = DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    gdf = cudf.DataFrame.from_pandas(df)

    assert_exceptions_equal(
        lfunc=df.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=([subset],),
        rfunc_args_and_kwargs=([subset],),
        compare_error_message=False,
    )


@pytest.mark.parametrize("keep", ["first", "last", False])
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4, 5, 6, 6],
        [],
        ["a", "b", "s", "sd", "a", "b"],
        Series(["aaa"] * 10, dtype="object"),
    ],
)
def test_drop_duplicates_series(data, keep):
    pds = _create_pandas_series(data)
    gds = cudf.from_pandas(pds)

    assert_df(pds.drop_duplicates(keep=keep), gds.drop_duplicates(keep=keep))
    pds.drop_duplicates(keep=keep, inplace=True)
    gds.drop_duplicates(keep=keep, inplace=True)
    assert_df(pds, gds)


def test_drop_duplicates():
    pdf = DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    # single column
    result = gdf.copy()
    result.drop_duplicates("AAA", inplace=True)
    expected = pdf.copy()
    expected.drop_duplicates("AAA", inplace=True)
    assert_df(result, expected)

    result = gdf.drop_duplicates("AAA", keep="last")
    expected = pdf.drop_duplicates("AAA", keep="last")
    assert_df(result, expected)

    result = gdf.drop_duplicates("AAA", keep=False)
    expected = pdf.drop_duplicates("AAA", keep=False)
    assert_df(result, expected)
    assert len(result) == 0

    # multi column
    expected = pdf.loc[[0, 1, 2, 3]]
    result = gdf.drop_duplicates(np.array(["AAA", "B"]))
    assert_df(result, expected)
    result = pdf.drop_duplicates(np.array(["AAA", "B"]))
    assert_df(result, expected)

    result = gdf.drop_duplicates(("AAA", "B"), keep="last")
    expected = pdf.drop_duplicates(("AAA", "B"), keep="last")
    assert_df(result, expected)

    result = gdf.drop_duplicates(("AAA", "B"), keep=False)
    expected = pdf.drop_duplicates(("AAA", "B"), keep=False)
    assert_df(result, expected)

    # consider everything
    df2 = gdf.loc[:, ["AAA", "B", "C"]]

    result = df2.drop_duplicates()
    # in this case only
    expected = df2.drop_duplicates(["AAA", "B"])
    assert_df(result, expected)

    result = df2.drop_duplicates(keep="last")
    expected = df2.drop_duplicates(["AAA", "B"], keep="last")
    assert_df(result, expected)

    result = df2.drop_duplicates(keep=False)
    expected = df2.drop_duplicates(["AAA", "B"], keep=False)
    assert_df(result, expected)

    # integers
    result = gdf.drop_duplicates("C")
    expected = pdf.drop_duplicates("C")
    assert_df(result, expected)
    result = gdf.drop_duplicates("C", keep="last")
    expected = pdf.drop_duplicates("C", keep="last")
    assert_df(result, expected)

    gdf["E"] = gdf["C"].astype("int8")
    result = gdf.drop_duplicates("E")
    pdf["E"] = pdf["C"].astype("int8")
    expected = pdf.drop_duplicates("E")
    assert_df(result, expected)
    result = gdf.drop_duplicates("E", keep="last")
    expected = pdf.drop_duplicates("E", keep="last")
    assert_df(result, expected)

    pdf = DataFrame({"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]})
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    pdf = DataFrame([[1, 0], [0, 2]])
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    pdf = DataFrame([[-2, 0], [0, -4]])
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    x = np.iinfo(np.int64).max / 3 * 2
    pdf = DataFrame([[-x, x], [0, x + 4]])
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    pdf = DataFrame([[-x, x], [x, x + 4]])
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    pdf = DataFrame([i] * 9 for i in range(16))
    pdf = pdf.append([[1] + [0] * 8], ignore_index=True)
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())


@pytest.mark.skip(reason="cudf does not support duplicate column names yet")
def test_drop_duplicates_with_duplicate_column_names():
    df = DataFrame([[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=["a", "a", "b"])
    df = cudf.DataFrame.from_pandas(df)

    result0 = df.drop_duplicates()
    assert_df(result0, df)

    result1 = df.drop_duplicates("a")
    expected1 = df[:2]
    assert_df(result1, expected1)


def test_drop_duplicates_for_take_all():
    pdf = DataFrame(
        {
            "AAA": ["foo", "bar", "baz", "bar", "foo", "bar", "qux", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    # single column
    result = gdf.drop_duplicates("AAA")
    expected = pdf.drop_duplicates("AAA")
    assert_df(result, expected)

    result = gdf.drop_duplicates("AAA", keep="last")
    expected = pdf.drop_duplicates("AAA", keep="last")
    assert_df(result, expected)

    result = gdf.drop_duplicates("AAA", keep=False)
    expected = pdf.drop_duplicates("AAA", keep=False)
    assert_df(result, expected)

    # multiple columns
    result = gdf.drop_duplicates(["AAA", "B"])
    expected = pdf.drop_duplicates(["AAA", "B"])
    assert_df(result, expected)

    result = gdf.drop_duplicates(["AAA", "B"], keep="last")
    expected = pdf.drop_duplicates(["AAA", "B"], keep="last")
    assert_df(result, expected)

    result = gdf.drop_duplicates(["AAA", "B"], keep=False)
    expected = pdf.drop_duplicates(["AAA", "B"], keep=False)
    assert_df(result, expected)


def test_drop_duplicates_tuple():
    pdf = DataFrame(
        {
            ("AA", "AB"): [
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "bar",
                "foo",
            ],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    # single column
    result = gdf.drop_duplicates(("AA", "AB"))
    expected = pdf.drop_duplicates(("AA", "AB"))
    assert_df(result, expected)

    result = gdf.drop_duplicates(("AA", "AB"), keep="last")
    expected = pdf.drop_duplicates(("AA", "AB"), keep="last")
    assert_df(result, expected)

    result = gdf.drop_duplicates(("AA", "AB"), keep=False)
    expected = pdf.drop_duplicates(("AA", "AB"), keep=False)  # empty df
    assert len(result) == 0
    assert_df(result, expected)

    # multi column
    expected = pdf.drop_duplicates((("AA", "AB"), "B"))
    result = gdf.drop_duplicates((("AA", "AB"), "B"))
    assert_df(result, expected)


@pytest.mark.parametrize(
    "df",
    [
        DataFrame(),
        DataFrame(columns=[]),
        DataFrame(columns=["A", "B", "C"]),
        DataFrame(index=[]),
        DataFrame(index=["A", "B", "C"]),
    ],
)
def test_drop_duplicates_empty(df):
    df = cudf.DataFrame.from_pandas(df)
    result = df.drop_duplicates()
    assert_df(result, df)

    result = df.copy()
    result.drop_duplicates(inplace=True)
    assert_df(result, df)


@pytest.mark.parametrize("num_columns", [3, 4, 5])
def test_dataframe_drop_duplicates_numeric_method(num_columns):
    comb = list(itertools.permutations(range(num_columns), num_columns))
    shuf = list(comb)
    random.Random(num_columns).shuffle(shuf)

    def get_pdf(n_dup):
        # create dataframe with n_dup duplicate rows
        rows = comb + shuf[:n_dup]
        random.Random(n_dup).shuffle(rows)
        return DataFrame(rows)

    for i in range(5):
        pdf = get_pdf(i)
        gdf = cudf.DataFrame.from_pandas(pdf)
        assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    # subset columns, single columns
    assert_df(
        gdf.drop_duplicates(pdf.columns[:-1]),
        pdf.drop_duplicates(pdf.columns[:-1]),
    )
    assert_df(
        gdf.drop_duplicates(pdf.columns[-1]),
        pdf.drop_duplicates(pdf.columns[-1]),
    )
    assert_df(
        gdf.drop_duplicates(pdf.columns[0]),
        pdf.drop_duplicates(pdf.columns[0]),
    )

    # subset columns shuffled
    cols = list(pdf.columns)
    random.Random(3).shuffle(cols)
    assert_df(gdf.drop_duplicates(cols), pdf.drop_duplicates(cols))
    random.Random(3).shuffle(cols)
    assert_df(gdf.drop_duplicates(cols[:-1]), pdf.drop_duplicates(cols[:-1]))
    random.Random(3).shuffle(cols)
    assert_df(gdf.drop_duplicates(cols[-1]), pdf.drop_duplicates(cols[-1]))
    assert_df(
        gdf.drop_duplicates(cols, keep="last"),
        pdf.drop_duplicates(cols, keep="last"),
    )


def test_dataframe_drop_duplicates_method():
    pdf = DataFrame(
        [(1, 2, "a"), (2, 3, "b"), (3, 4, "c"), (2, 3, "d"), (3, 5, "c")],
        columns=["n1", "n2", "s1"],
    )
    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates(), pdf.drop_duplicates())

    assert_eq(
        gdf.drop_duplicates("n1")["n1"].reset_index(drop=True),
        pdf.drop_duplicates("n1")["n1"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("n2")["n2"].reset_index(drop=True),
        pdf.drop_duplicates("n2")["n2"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("s1")["s1"].reset_index(drop=True),
        pdf.drop_duplicates("s1")["s1"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("s1", keep="last")["s1"]
        .sort_index()
        .reset_index(drop=True),
        pdf.drop_duplicates("s1", keep="last")["s1"].reset_index(drop=True),
    )
    assert gdf.drop_duplicates("s1", inplace=True) is None

    gdf = cudf.DataFrame.from_pandas(pdf)
    assert_df(gdf.drop_duplicates("n1"), pdf.drop_duplicates("n1"))
    assert_df(gdf.drop_duplicates("n2"), pdf.drop_duplicates("n2"))
    assert_df(gdf.drop_duplicates("s1"), pdf.drop_duplicates("s1"))
    assert_df(
        gdf.drop_duplicates(["n1", "n2"]), pdf.drop_duplicates(["n1", "n2"])
    )
    assert_df(
        gdf.drop_duplicates(["n1", "s1"]), pdf.drop_duplicates(["n1", "s1"])
    )

    # Test drop error
    assert_exceptions_equal(
        lfunc=pdf.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=(["n3"],),
        rfunc_args_and_kwargs=(["n3"],),
        expected_error_message="columns {'n3'} do not exist",
    )

    assert_exceptions_equal(
        lfunc=pdf.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=([["n1", "n4", "n3"]],),
        rfunc_args_and_kwargs=([["n1", "n4", "n3"]],),
        expected_error_message="columns {'n[34]', 'n[34]'} do not exist",
    )


def test_datetime_drop_duplicates():

    date_df = cudf.DataFrame()
    date_df["date"] = date_range("11/20/2018", periods=6, freq="D")
    date_df["value"] = np.random.sample(len(date_df))

    df = concat([date_df, date_df[:4]])
    assert_df(df[:-4], df.drop_duplicates())

    df2 = df.reset_index()
    assert_df(df2[:-4], df2.drop_duplicates())

    df3 = df.set_index("date")
    assert_df(df3[:-4], df3.drop_duplicates())


def test_drop_duplicates_NA():
    # none
    df = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    df = cudf.DataFrame.from_pandas(df)
    # single column
    result = df.drop_duplicates("A")
    expected = df.to_pandas().loc[[0, 2, 3]]
    assert_df(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.to_pandas().loc[[1, 6, 7]]
    assert_df(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.to_pandas().loc[[]]  # empty df
    assert_df(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["A", "B"])
    expected = df.to_pandas().loc[[0, 2, 3, 6]]
    assert_df(result, expected)

    result = df.drop_duplicates(["A", "B"], keep="last")
    expected = df.to_pandas().loc[[1, 5, 6, 7]]
    assert_df(result, expected)

    result = df.drop_duplicates(["A", "B"], keep=False)
    expected = df.to_pandas().loc[[6]]
    assert_df(result, expected)

    # nan
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    df = cudf.DataFrame.from_pandas(df)
    # single column
    result = df.drop_duplicates("C")
    expected = df[:2]
    assert_df(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.to_pandas().loc[[3, 7]]
    assert_df(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.to_pandas().loc[[]]  # empty df
    assert_df(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["C", "B"])
    expected = df.to_pandas().loc[[0, 1, 2, 4]]
    assert_df(result, expected)

    result = df.drop_duplicates(["C", "B"], keep="last")
    expected = df.to_pandas().loc[[1, 3, 6, 7]]
    assert_df(result, expected)

    result = df.drop_duplicates(["C", "B"], keep=False)
    expected = df.to_pandas().loc[[1]]
    assert_df(result, expected)


def test_drop_duplicates_NA_for_take_all():
    # TODO: PANDAS 1.0 support - add ignore_index for
    # pandas drop_duplicates calls in this function.

    # none
    pdf = DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "baz", "bar", "qux"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0],
        }
    )

    df = cudf.DataFrame.from_pandas(pdf)
    # single column
    result = df.drop_duplicates("A")
    expected = pdf.iloc[[0, 2, 3, 5, 7]]
    assert_df(result, expected)
    assert_df(
        df.drop_duplicates("A", ignore_index=True),
        result.reset_index(drop=True),
    )

    result = df.drop_duplicates("A", keep="last")
    expected = pdf.iloc[[1, 4, 5, 6, 7]]
    assert_df(result, expected)
    assert_df(
        df.drop_duplicates("A", ignore_index=True, keep="last"),
        result.reset_index(drop=True),
    )

    result = df.drop_duplicates("A", keep=False)
    expected = pdf.iloc[[5, 7]]
    assert_df(result, expected)
    assert_df(
        df.drop_duplicates("A", ignore_index=True, keep=False),
        result.reset_index(drop=True),
    )

    # nan

    # single column
    result = df.drop_duplicates("C")
    expected = pdf.iloc[[0, 1, 5, 6]]
    assert_df(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = pdf.iloc[[3, 5, 6, 7]]
    assert_df(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = pdf.iloc[[5, 6]]
    assert_df(result, expected)


def test_drop_duplicates_inplace():
    orig = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    orig = cudf.DataFrame.from_pandas(orig)
    # single column
    df = orig.copy()
    df.drop_duplicates("A", inplace=True)
    expected = orig[:2]
    result = df
    assert_df(result, expected)

    df = orig.copy()
    df.drop_duplicates("A", keep="last", inplace=True)
    expected = orig.loc[[6, 7]]
    result = df
    assert_df(result, expected)

    df = orig.copy()
    df.drop_duplicates("A", keep=False, inplace=True)
    expected = orig.loc[[]]
    result = df
    assert_df(result, expected)
    assert len(df) == 0

    # multi column
    df = orig.copy()
    df.drop_duplicates(["A", "B"], inplace=True)
    expected = orig.loc[[0, 1, 2, 3]]
    result = df
    assert_df(result, expected)

    df = orig.copy()
    df.drop_duplicates(["A", "B"], keep="last", inplace=True)
    expected = orig.loc[[0, 5, 6, 7]]
    result = df
    assert_df(result, expected)

    df = orig.copy()
    df.drop_duplicates(["A", "B"], keep=False, inplace=True)
    expected = orig.loc[[0]]
    result = df
    assert_df(result, expected)

    # consider everything
    orig2 = orig.loc[:, ["A", "B", "C"]].copy()

    df2 = orig2.copy()
    df2.drop_duplicates(inplace=True)
    # in this case only
    expected = orig2.drop_duplicates(["A", "B"])
    result = df2
    assert_df(result, expected)

    df2 = orig2.copy()
    df2.drop_duplicates(keep="last", inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep="last")
    result = df2
    assert_df(result, expected)

    df2 = orig2.copy()
    df2.drop_duplicates(keep=False, inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep=False)
    result = df2
    assert_df(result, expected)


def test_drop_duplicates_multi_index():
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]

    idx = MultiIndex.from_tuples(list(zip(*arrays)), names=["a", "b"])
    pdf = DataFrame(np.random.randint(0, 2, (8, 4)), index=idx)
    gdf = cudf.DataFrame.from_pandas(pdf)

    expected = pdf.drop_duplicates()
    result = gdf.drop_duplicates()
    assert_df(result.to_pandas(), expected)
    # FIXME: to_pandas needed until sort_index support for MultiIndex

    for col in gdf.columns:
        assert_df(
            gdf[col].drop_duplicates().to_pandas(),
            pdf[col].drop_duplicates(),
        )
