import pytest

import datetime as dt
import numpy as np
import pandas as pd
from pandas.util.testing import (
    assert_index_equal, assert_series_equal,
    assert_frame_equal
)
from cudf.dataframe import Series, DataFrame
from cudf.dataframe.index import DatetimeIndex
from cudf.tests.utils import assert_eq


def data1():
    return pd.date_range('20010101', '20020215', freq='400h')


def data2():
    return pd.date_range('20010101', '20020215',
                         freq='400h',
                         tz='US/Eastern')


def numerical_data():
    return np.arange(1, 10)


fields = ['year', 'month', 'day',
          'hour', 'minute', 'second']


@pytest.mark.parametrize('data', [data1(), data2()])
def test_series(data):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    np.testing.assert_equal(
        np.array(pd_data),
        np.array(gdf_data),
        )


# libgdf doesn't respect timezones
@pytest.mark.parametrize('data', [data1()])
@pytest.mark.parametrize('field', fields)
def test_dt_series(data, field):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    base = getattr(pd_data.dt, field)
    test = getattr(gdf_data.dt, field).to_pandas().astype('int64')
    assert_series_equal(base, test)


@pytest.mark.parametrize('data', [data1()])
@pytest.mark.parametrize('field', fields)
def test_dt_index(data, field):
    pd_data = data.copy()
    gdf_data = DatetimeIndex(pd_data)
    assert_index_equal(getattr(gdf_data, field).to_pandas(),
                       getattr(pd_data, field))


def test_setitem_datetime():
    a = DataFrame()
    a['a'] = pd.date_range('20010101', '20010105').values
    # TODO check some stuff


def test_issue_165():
    df_pandas = pd.DataFrame()
    start_date = dt.datetime.strptime("2000-10-21", '%Y-%m-%d')
    data = [(start_date + dt.timedelta(days=x)) for x in range(6)]
    df_pandas["dates"] = data
    df_pandas["num"] = [1, 2, 3, 4, 5, 6]
    df_cudf = DataFrame.from_pandas(df_pandas)

    base = df_pandas.query("dates==@start_date")
    test = df_cudf.query("dates==@start_date")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date
    base_mask = df_pandas.dates == start_date
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_ts = pd.Timestamp(start_date)
    test = df_cudf.query("dates==@start_date_ts")
    base = df_pandas.query("dates==@start_date_ts")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date_ts
    base_mask = df_pandas.dates == start_date_ts
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_np = np.datetime64(start_date_ts, 'ns')
    test = df_cudf.query("dates==@start_date_np")
    base = df_pandas.query("dates==@start_date_np")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_cudf.dates == start_date_np
    base_mask = df_pandas.dates == start_date_np
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0


@pytest.mark.parametrize('data', [data1(), data2()])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32',
                                   'int64', 'float32', 'float64'])
def test_typecast_from_datetime(data, dtype):
    pd_data = pd.Series(data.copy())
    np_data = np.array(pd_data).astype('datetime64[ms]')
    gdf_data = Series(pd_data)

    np_casted = np_data.astype(dtype)
    gdf_casted = gdf_data.astype(dtype)

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize('data', [numerical_data()])
@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32',
                                   'int64', 'float32', 'float64'])
def test_typecast_to_datetime(data, dtype):
    np_data = data.astype(dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype('datetime64[ms]')
    gdf_casted = gdf_data.astype('datetime64[ms]')

    np.testing.assert_equal(np_casted, np.array(gdf_casted))


@pytest.mark.parametrize('data', [numerical_data()])
@pytest.mark.parametrize('nulls', ['some', 'all'])
def test_to_from_pandas_nulls(data, nulls):
    pd_data = pd.Series(data.copy().astype('datetime64[ms]'))
    if nulls == 'some':
        # Fill half the values with NaT
        pd_data[list(range(0, len(pd_data), 2))] = np.datetime64('nat')
    elif nulls == 'all':
        # Fill all the values with NaT
        pd_data[:] = np.datetime64('nat')
    gdf_data = Series.from_pandas(pd_data)

    expect = pd_data
    got = gdf_data.to_pandas()

    assert_eq(expect, got)
