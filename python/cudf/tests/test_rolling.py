import pandas as pd
import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    'data',
    [
        [],
        [1, 1, 1, 1],
        [1, 2, 3, 4],
        [1, 2, 4, 9, 9, 4]
    ]
)
@pytest.mark.parametrize(
    'agg',
    ['sum', 'min', 'max', 'mean', 'count']
)
@pytest.mark.parametrize(
    'nulls',
    ['none', 'one', 'some', 'all']
)
def test_rollling_series_basic(data, agg, nulls):
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == 'one':
            p = np.random.randint(0, len(data))
            psr[p] = None
        elif nulls == 'some':
            p1, p2 = np.random.randint(0, len(data), (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == 'all':
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    for window_size in range(1, len(data)+1):
        for min_periods in range(1, window_size+1):
            assert_eq(
                getattr(psr.rolling(window_size, min_periods), agg)().fillna(-1),
                getattr(gsr.rolling(window_size, min_periods), agg)().fillna(-1),
                check_dtype=False
            )

@pytest.mark.parametrize(
    'data',
    [
        {'a': [], 'b': []},
        {'a': [1, 2, 3, 4], 'b': [1, 2, 3, 4]},
        {'a': [1, 2, 4, 9, 9, 4], 'b': [1, 2, 4, 9, 9, 4]},
        {'a': np.array([1, 2, 4, 9, 9, 4]), 'b': np.array([1.5, 2.2, 2.2, 8.0, 9.1, 4.2])}
    ]
)
@pytest.mark.parametrize(
    'agg',
    ['sum', 'min', 'max', 'mean', 'count']
)
@pytest.mark.parametrize(
    'nulls',
    ['none', 'one', 'some', 'all']
)

def test_rolling_dataframe_basic(data, agg, nulls):
    pdf = pd.DataFrame(data)

    if len(pdf) > 0:
        for col_name in pdf.columns:
            if nulls == 'one':
                p = np.random.randint(0, len(data))
                pdf[col_name][p] = None
            elif nulls == 'some':
                p1, p2 = np.random.randint(0, len(data), (2,))
                pdf[col_name][p1] = None
                pdf[col_name][p2] = None
            elif nulls == 'all':
                pdf[col_name][:] = None

    gdf = cudf.from_pandas(pdf)

    for window_size in range(1, len(data)+1):
        for min_periods in range(1, window_size+1):
            assert_eq(
                getattr(pdf.rolling(window_size, min_periods), agg)().fillna(-1),
                getattr(gdf.rolling(window_size, min_periods), agg)().fillna(-1),
                check_dtype=False
            )
