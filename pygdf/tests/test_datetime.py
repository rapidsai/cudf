import pytest

import datetime as dt
import numpy as np
import pandas as pd
from pandas.util.testing import (
    assert_index_equal, assert_series_equal,
    assert_frame_equal
)
from pygdf.dataframe import Series, DataFrame
from pygdf.index import DatetimeIndex


def data1():
    return pd.date_range('20010101', '20020215', freq='400h')


def data2():
    return pd.date_range('20010101', '20020215',
                         freq='400h',
                         tz='US/Eastern')


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
    df_pygdf = DataFrame.from_pandas(df_pandas)

    base = df_pandas.query("dates==@start_date")
    test = df_pygdf.query("dates==@start_date")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0


    mask = df_pygdf.dates == start_date
    base_mask = df_pandas.dates == start_date
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_ts = pd.Timestamp(start_date)
    test = df_pygdf.query("dates==@start_date_ts")
    base = df_pandas.query("dates==@start_date_ts")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_pygdf.dates == start_date_ts
    base_mask = df_pandas.dates == start_date_ts
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0

    start_date_np = np.datetime64(start_date_ts, 'ns')
    test = df_pygdf.query("dates==@start_date_np")
    base = df_pandas.query("dates==@start_date_np")
    assert_frame_equal(base, test.to_pandas())
    assert len(test) > 0

    mask = df_pygdf.dates == start_date_np
    base_mask = df_pandas.dates == start_date_np
    assert_series_equal(mask.to_pandas(), base_mask, check_names=False)
    assert mask.to_pandas().sum() > 0
