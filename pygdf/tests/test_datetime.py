import pytest
import pandas as pd

from pygdf.dataframe import Series, Index, DataFrame


def data1():
    return pd.date_range('20010101', '20020215', freq='400h')


def data2():
    return pd.date_range('20010101', '20020215',
                         freq='400h',
                         tz='US/Eastern')


fields = ['year', 'month', 'day',
          'date', 'hour', 'minute']


@pytest.mark.parametrize('data', [data1(), data2()])
@pytest.mark.parametrize('field', fields)
def test_dt_series(data, field):
    pd_data = pd.Series(data.copy())
    gdf_data = Series(pd_data)
    np.testing.assert_equal(
        getattr(gdf_data.dt, field).to_array(),
        getattr(pd_data.dt, field).values
    )


@pytest.mark.parametrize('data', [data1(), data2()])
@pytest.mark.parametrize('field', fields)
def test_dt_index(data, field):
    pd_data = data.copy()
    gdf_data = Index(pd_data)
    np.testing.assert_equal(
        getattr(gdf_data, field).to_array(),
        getattr(pd_data, field).values
    )


def test_setitem_datetime():
    a = DataFrame()
    a['a'] = pd.date_range('20010101', '20010105').values
    # TODO check some stuff
