import pytest
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq

def test_dataset_timeseries():
    a = cudf.datasets.timeseries(dtypes={"x": int, "y": float}, freq="120s", seed=1)
    b = cudf.datasets.timeseries(dtypes={"x": int, "y": float}, freq="120s", seed=1)

    assert_eq(a, b)

    assert a['x'].head().dtype == int
    assert a['y'].head().dtype == float
    assert a.index.name == 'timestamp'

    cdf = cudf.datasets.make_timeseries('2000', '2010',
                                       {'value': float, 'name': 'category', 'id': int},
                                       freq='2H', partition_freq='1D', seed=1)

    assert cdf['value'].head().dtype == float
    assert cdf['id'].head().dtype == int
    assert cdf['name'].head().dtype == pd.api.types.CategoricalDtype()