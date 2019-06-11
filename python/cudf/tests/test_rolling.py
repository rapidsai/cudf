import pandas as pd
import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

@pytest.mark.parametrize(
    'data',
    [
        [],
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
        assert_eq(
            getattr(psr.rolling(window_size), agg)().fillna(-1),
            getattr(gsr.rolling(window_size), agg)().fillna(-1),
            check_dtype=False
        )
