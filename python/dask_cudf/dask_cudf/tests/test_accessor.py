import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from dask import dataframe as dd

import dask_cudf as dgd

from cudf import DataFrame, Series
from cudf.tests.utils import assert_eq

#############################################################################
#                        Datetime Accessor                                  #
#############################################################################


def data_dt_1():
    return pd.date_range("20010101", "20020215", freq="400h")


def data_dt_2():
    return np.random.randn(100)


dt_fields = ["year", "month", "day", "hour", "minute", "second"]


@pytest.mark.parametrize("data", [data_dt_2()])
def test_datetime_accessor_initialization(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)
    with pytest.raises(AttributeError):
        dsr.dt


@pytest.mark.parametrize("data", [data_dt_1()])
def test_series(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)

    np.testing.assert_equal(np.array(pdsr), dsr.compute().to_array())


@pytest.mark.parametrize("data", [data_dt_1()])
@pytest.mark.parametrize("field", dt_fields)
def test_dt_series(data, field):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)
    base = getattr(pdsr.dt, field)
    test = getattr(dsr.dt, field).compute().to_pandas().astype("int64")
    assert_series_equal(base, test)


@pytest.mark.parametrize("data", [data_dt_1()])
def test_dt_accessor(data):
    df = DataFrame({"dt_col": data.copy()})
    ddf = dgd.from_cudf(df, npartitions=5)

    for i in ["year", "month", "day", "hour", "minute", "second", "weekday"]:
        assert i in dir(ddf.dt_col.dt)
        assert_series_equal(
            getattr(ddf.dt_col.dt, i).compute().to_pandas(),
            getattr(df.dt_col.dt, i).to_pandas(),
        )


#############################################################################
#                        Categorical Accessor                               #
#############################################################################


def data_cat_1():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    return cat


def data_cat_2():
    return pd.Series([1, 2, 3])


def data_cat_3():
    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    return cat1, cat2


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_accessor_initialization1(data):
    sr = Series(data.copy())
    dsr = dgd.from_cudf(sr, npartitions=5)
    dsr.cat


@pytest.mark.parametrize("data", [data_cat_2()])
def test_categorical_accessor_initialization2(data):
    sr = Series(data.copy())
    dsr = dgd.from_cudf(sr, npartitions=5)
    with pytest.raises(AttributeError):
        dsr.cat


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_basic(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dgd.from_cudf(sr, npartitions=2)
    result = dsr.compute()
    np.testing.assert_array_equal(cat.codes, result.to_array())

    assert dsr.dtype.to_pandas() == pdsr.dtype
    # Test attributes
    assert pdsr.cat.ordered == dsr.cat.ordered

    assert_eq(pdsr.cat.categories, dsr.cat.categories)

    np.testing.assert_array_equal(pdsr.cat.codes.values, result.to_array())

    string = str(result)
    expect_str = """
0 a
1 a
2 b
3 c
4 a
"""
    assert all(x == y for x, y in zip(string.split(), expect_str.split()))

    df = DataFrame()
    df["a"] = ["xyz", "abc", "def"] * 10

    pdf = df.to_pandas()
    cddf = dgd.from_cudf(df, 1)
    cddf["b"] = cddf["a"].astype("category")

    ddf = dd.from_pandas(pdf, 1)
    ddf["b"] = ddf["a"].astype("category")

    assert_eq(ddf._meta_nonempty["b"], cddf._meta_nonempty["b"])

    with pytest.raises(NotImplementedError):
        cddf["b"].cat.categories

    with pytest.raises(NotImplementedError):
        ddf["b"].cat.categories

    cddf = cddf.categorize()
    ddf = ddf.categorize()

    assert_eq(ddf["b"].cat.categories, cddf["b"].cat.categories)
    assert_eq(ddf["b"].cat.ordered, cddf["b"].cat.ordered)


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_compare_unordered(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dgd.from_cudf(sr, npartitions=2)

    # Test equality
    out = dsr == dsr
    assert out.dtype == np.bool_
    assert np.all(out.compute())
    assert np.all(pdsr == pdsr)

    # Test inequality
    out = dsr != dsr
    assert not np.any(out.compute())
    assert not np.any(pdsr != pdsr)

    assert not dsr.cat.ordered
    assert not pdsr.cat.ordered

    with pytest.raises(
        (TypeError, ValueError),
        match="Unordered Categoricals can only compare equality or not",
    ):
        pdsr < pdsr

    with pytest.raises(
        (TypeError, ValueError),
        match="Unordered Categoricals can only compare equality or not",
    ):
        dsr < dsr


@pytest.mark.parametrize("data", [data_cat_3()])
def test_categorical_compare_ordered(data):
    cat1 = data[0]
    cat2 = data[1]
    pdsr1 = pd.Series(cat1)
    pdsr2 = pd.Series(cat2)
    sr1 = Series(cat1)
    sr2 = Series(cat2)
    dsr1 = dgd.from_cudf(sr1, npartitions=2)
    dsr2 = dgd.from_cudf(sr2, npartitions=2)

    # Test equality
    out = dsr1 == dsr1
    assert out.dtype == np.bool_
    assert np.all(out.compute().to_array())
    assert np.all(pdsr1 == pdsr1)

    # Test inequality
    out = dsr1 != dsr1
    assert not np.any(out.compute().to_array())
    assert not np.any(pdsr1 != pdsr1)

    assert dsr1.cat.ordered
    assert pdsr1.cat.ordered

    # Test ordered operators
    np.testing.assert_array_equal(
        pdsr1 < pdsr2, (dsr1 < dsr2).compute().to_array()
    )
    np.testing.assert_array_equal(
        pdsr1 > pdsr2, (dsr1 > dsr2).compute().to_array()
    )


#############################################################################
#                        String Accessor                                    #
#############################################################################


def data_str_1():
    return pd.Series(["20190101", "20190102", "20190103"])


@pytest.mark.parametrize("data", [data_str_1()])
def test_string_slicing(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=2)
    base = pdsr.str.slice(0, 4)
    test = dsr.str.slice(0, 4).compute()
    assert_eq(base, test)


def test_categorical_categories():

    df = DataFrame(
        {"a": ["a", "b", "c", "d", "e", "e", "a", "d"], "b": range(8)}
    )
    df["a"] = df["a"].astype("category")
    pdf = df.to_pandas(nullable_pd_dtype=False)

    ddf = dgd.from_cudf(df, 2)
    dpdf = dd.from_pandas(pdf, 2)

    dd.assert_eq(
        ddf.a.cat.categories.to_series().to_pandas(nullable_pd_dtype=False),
        dpdf.a.cat.categories.to_series(),
        check_index=False,
    )


def test_categorical_as_known():
    df = dgd.from_cudf(DataFrame({"col_1": [0, 1, 2, 3]}), npartitions=2)
    df["col_1"] = df["col_1"].astype("category")
    actual = df["col_1"].cat.as_known()

    pdf = dd.from_pandas(pd.DataFrame({"col_1": [0, 1, 2, 3]}), npartitions=2)
    pdf["col_1"] = pdf["col_1"].astype("category")
    expected = pdf["col_1"].cat.as_known()
    dd.assert_eq(expected, actual)


def test_str_slice():

    df = DataFrame({"a": ["abc,def,123", "xyz,hi,bye"]})

    ddf = dgd.from_cudf(df, 1)
    pdf = df.to_pandas()

    dd.assert_eq(
        pdf.a.str.split(",", expand=True, n=1),
        ddf.a.str.split(",", expand=True, n=1),
    )
    dd.assert_eq(
        pdf.a.str.split(",", expand=True, n=2),
        ddf.a.str.split(",", expand=True, n=2),
    )
