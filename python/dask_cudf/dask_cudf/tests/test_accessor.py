import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_series_equal

from cudf.dataframe import Series

import dask_cudf as dgd

#############################################################################
#                        Datetime Accessor                                  #
#############################################################################


def data_dt_1():
    return pd.date_range("20010101", "20020215", freq="400h")


def data_dt_2():
    return np.random.randn(100)


dt_fields = ["year", "month", "day", "hour", "minute", "second"]


@pytest.mark.parametrize("data", [data_dt_2()])
@pytest.mark.xfail(raises=AttributeError)
def test_datetime_accessor_initialization(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)
    dsr.dt


@pytest.mark.parametrize("data", [data_dt_1()])
def test_series(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)

    np.testing.assert_equal(np.array(pdsr), np.array(dsr.compute()))


@pytest.mark.parametrize("data", [data_dt_1()])
@pytest.mark.parametrize("field", dt_fields)
def test_dt_series(data, field):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dgd.from_cudf(sr, npartitions=5)
    base = getattr(pdsr.dt, field)
    test = getattr(dsr.dt, field).compute().to_pandas().astype("int64")
    assert_series_equal(base, test)


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


@pytest.mark.parametrize("data", [data_cat_2()])
@pytest.mark.xfail(raises=AttributeError)
def test_categorical_accessor_initialization(data):
    sr = Series(data.copy())
    dsr = dgd.from_cudf(sr, npartitions=5)
    dsr.cat


@pytest.mark.xfail(reason="")
@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_basic(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dgd.from_cudf(sr, npartitions=2)
    result = dsr.compute()
    np.testing.assert_array_equal(cat.codes, result.to_array())
    assert dsr.dtype == pdsr.dtype

    # Test attributes
    assert pdsr.cat.ordered == dsr.cat.ordered
    # TODO: Investigate dsr.cat.categories: It raises
    # ValueError: Expected iterable of tuples of (name, dtype),
    # got ('a', 'b', 'c')
    # assert(tuple(pdsr.cat.categories) == tuple(dsr.cat.categories))

    np.testing.assert_array_equal(pdsr.cat.codes.data, result.to_array())
    np.testing.assert_array_equal(pdsr.cat.codes.dtype, dsr.cat.codes.dtype)

    string = str(result)
    expect_str = """
0 a
1 a
2 b
3 c
4 a
"""
    assert all(x == y for x, y in zip(string.split(), expect_str.split()))


@pytest.mark.xfail(reason="")
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

    with pytest.raises((TypeError, ValueError)) as raises:
        pdsr < pdsr

    raises.match("Unordered Categoricals can only compare equality or not")

    with pytest.raises((TypeError, ValueError)) as raises:
        dsr < dsr

    raises.match("Unordered Categoricals can only compare equality or not")


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
    np.testing.assert_array_equal(pdsr1 < pdsr2, (dsr1 < dsr2).compute())
    np.testing.assert_array_equal(pdsr1 > pdsr2, (dsr1 > dsr2).compute())
