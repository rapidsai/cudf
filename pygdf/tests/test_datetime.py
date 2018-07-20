import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_index_equal
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

# fields = ['year']


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
    base = getattr(pd_data.dt, field).values
    test = getattr(gdf_data.dt, field).to_array()
    np.testing.assert_equal(base, test)


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
