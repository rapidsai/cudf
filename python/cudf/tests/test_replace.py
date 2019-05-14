import pytest

import numpy as np
import pandas as pd

from cudf.dataframe import Series, DataFrame
from cudf.tests.utils import assert_eq


def test_series_replace():
    a1 = np.array([0, 1, 2, 3, 4])

    # Numerical
    a2 = np.array([5, 1, 2, 3, 4])
    sr1 = Series(a1)
    sr2 = sr1.replace(0, 5)
    np.testing.assert_equal(sr2.to_array(), a2)

    # Categorical
    psr3 = pd.Series(["one", "two", "three"], dtype='category')
    psr4 = psr3.replace("one", "two")
    sr3 = Series.from_pandas(psr3)
    sr4 = sr3.replace("one", "two")
    pd.testing.assert_series_equal(sr4.to_pandas(), psr4)

    # List input
    a6 = np.array([5, 6, 2, 3, 4])
    sr6 = sr1.replace([0, 1], [5, 6])
    np.testing.assert_equal(sr6.to_array(), a6)

    a7 = np.array([5.5, 6.5, 2, 3, 4])
    sr7 = sr1.replace([0, 1], [5.5, 6.5])
    np.testing.assert_equal(sr7.to_array(), a7)

    # Series input
    a8 = np.array([5, 5, 5, 3, 4])
    sr8 = sr1.replace(sr1[:3], 5)
    np.testing.assert_equal(sr8.to_array(), a8)


def test_dataframe_replace():
    # numerical
    pdf1 = pd.DataFrame({'a': [0, 1, 2, 3], 'b': [0, 1, 2, 3]})
    gdf1 = DataFrame.from_pandas(pdf1)
    pdf2 = pdf1.replace(0, 4)
    gdf2 = gdf1.replace(0, 4)
    pd.testing.assert_frame_equal(gdf2.to_pandas(), pdf2)

    # categorical
    pdf4 = pd.DataFrame({'a': ['one', 'two', 'three'],
                         'b': ['one', 'two', 'three']}, dtype='category')
    gdf4 = DataFrame.from_pandas(pdf4)
    pdf5 = pdf4.replace('two', 'three')
    gdf5 = gdf4.replace('two', 'three')
    pd.testing.assert_frame_equal(gdf5.to_pandas(), pdf5)

    # list input
    pdf6 = pdf1.replace([0, 1], [4, 5])
    gdf6 = gdf1.replace([0, 1], [4, 5])
    pd.testing.assert_frame_equal(gdf6.to_pandas(), pdf6)

    pdf7 = pdf1.replace([0, 1], 4)
    gdf7 = gdf1.replace([0, 1], 4)
    pd.testing.assert_frame_equal(gdf7.to_pandas(), pdf7)

    # dict input:
    pdf8 = pdf1.replace({'a': 0, 'b': 0}, {'a': 4, 'b': 5})
    gdf8 = gdf1.replace({'a': 0, 'b': 0}, {'a': 4, 'b': 5})
    pd.testing.assert_frame_equal(gdf8.to_pandas(), pdf8)

    pdf9 = pdf1.replace({'a': 0}, {'a': 4})
    gdf9 = gdf1.replace({'a': 0}, {'a': 4})
    pd.testing.assert_frame_equal(gdf9.to_pandas(), pdf9)


@pytest.mark.parametrize('data_dtype', ['int8', 'int16', 'int32', 'int64',
                                        'float32', 'float64'])
@pytest.mark.parametrize('fill_dtype', ['int8', 'int16', 'int32', 'int64',
                                        'float32', 'float64'])
@pytest.mark.parametrize(
    'fill_type',
    ['scalar', 'series'])
@pytest.mark.parametrize(
    'null_value',
    [None, np.nan])
@pytest.mark.parametrize(
    'inplace',
    [True, False])
def test_series_fillna_numerical(data_dtype, fill_dtype,
                                 fill_type, null_value, inplace):
    # TODO: These tests should use Pandas' nullable int type
    # when we support a recent enough version of Pandas
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html

    if fill_type == 'scalar':
        fill_value = np.random.randint(0, 5)
        expect = np.array(
            [0, 1, fill_value, 2, fill_value],
            dtype=data_dtype)
    elif fill_type == 'series':
        data = np.random.randint(0, 5, (5,))
        fill_value = pd.Series(data, dtype=data_dtype)
        expect = np.array(
            [0, 1, fill_value[2], 2, fill_value[4]],
            dtype=data_dtype)

    sr = Series([0, 1, null_value, 2, null_value], dtype=data_dtype)
    result = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        result = sr

    got = result.to_array()

    np.testing.assert_equal(expect, got)


@pytest.mark.parametrize(
    'fill_type',
    ['scalar', 'series'])
@pytest.mark.parametrize(
    'null_value',
    [None, np.nan])
@pytest.mark.parametrize(
    'inplace',
    [True, False])
def test_fillna_categorical(fill_type, null_value, inplace):
    data = pd.Series(['a', 'b', 'a', null_value, 'c', null_value],
                     dtype='category')
    sr = Series.from_pandas(data)

    if fill_type == 'scalar':
        fill_value = 'c'
        expect = pd.Series(['a', 'b', 'a', 'c', 'c', 'c'],
                           dtype='category')
    elif fill_type == 'series':
        fill_value = pd.Series(['c', 'c', 'c', 'c', 'c', 'a'],
                               dtype='category')
        expect = pd.Series(['a', 'b', 'a', 'c', 'c', 'a'],
                           dtype='category')

    got = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


@pytest.mark.parametrize(
    'fill_type',
    ['scalar', 'series'])
@pytest.mark.parametrize(
    'inplace',
    [True, False])
def test_fillna_datetime(fill_type, inplace):
    psr = pd.Series(pd.date_range('2010-01-01', '2020-01-10', freq='1y'))

    if fill_type == 'scalar':
        fill_value = pd.Timestamp('2010-01-02')
    elif fill_type == 'series':
        fill_value = psr + pd.Timedelta('1d')

    psr[[5, 9]] = None
    sr = Series.from_pandas(psr)

    expect = psr.fillna(fill_value)
    got = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


@pytest.mark.parametrize(
    'fill_type',
    ['scalar', 'series', 'dict'])
@pytest.mark.parametrize(
    'inplace',
    [True, False])
def test_fillna_dataframe(fill_type, inplace):
    pdf = pd.DataFrame({'a': [1, 2, None], 'b': [None, None, 5]})
    gdf = DataFrame.from_pandas(pdf)

    if fill_type == 'scalar':
        fill_value = 5
    elif fill_type == 'series':
        fill_value = Series([3, 4, 5])
    else:
        fill_value = {'a': 5, 'b': Series([3, 4, 5])}

    expect = pdf.fillna(fill_value)
    got = gdf.fillna(fill_value, inplace=inplace)

    if inplace:
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize(
    'fill_type',
    ['scalar', 'series'])
@pytest.mark.parametrize(
    'inplace',
    [True, False])
def test_fillna_string(fill_type, inplace):
    psr = pd.Series(['z', None, 'z', None])

    if fill_type == 'scalar':
        fill_value = 'a'
    elif fill_type == 'series':
        fill_value = Series(['a', 'b', 'c', 'd'])

    sr = Series.from_pandas(psr)

    expect = psr.fillna(fill_value)
    got = sr.fillna(fill_value, inplace=inplace)

    if inplace:
        got = sr

    assert_eq(expect, got)


@pytest.mark.parametrize(
    'data_dtype',
    ['int8', 'int16', 'int32', 'int64'])
def test_series_fillna_invalid_dtype(data_dtype):
    gdf = Series([1, 2, None, 3], dtype=data_dtype)
    fill_value = 2.5
    with pytest.raises(TypeError) as raises:
        gdf.fillna(fill_value)
    raises.match("Cannot safely cast non-equivalent {} to {}".format(
        np.dtype(type(fill_value)).type.__name__,
        gdf.dtype.type.__name__
    ))
