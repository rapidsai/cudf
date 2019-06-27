import pytest

import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    'data',
    [
        [],
        [1.0, 2, None, 4],
        ['one', 'two', 'three', 'four'],
        pd.Series(['a', 'b', 'c', 'd'], dtype='category'),
        pd.Series(pd.date_range('2010-01-01', '2010-01-04'))
    ]
)
@pytest.mark.parametrize(
    'nulls',
    ['one', 'some', 'all', 'none']
)
def test_dropna(data, nulls):

    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == 'one':
            p = np.random.randint(0, 4)
            psr[p] = None
        elif nulls == 'some':
            p1, p2 = np.random.randint(0, 4, (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == 'all':
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    check_dtype = True
    if gsr.null_count == len(gsr):
        check_dtype = False

    assert_eq(psr.dropna(), gsr.dropna(), check_dtype=check_dtype)
