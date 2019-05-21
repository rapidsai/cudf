import pytest

import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    'data',
    [
        [1.0, 2, None, 4],
        ['one', 'two', None, 'four'],
        [None, None, None, None],
        [1.0, 2, 3, 4],
        []
    ]
)
def test_dropna(data):
    gs = cudf.Series(data)
    ps = pd.Series(data)

    check_dtype = True
    if gs.null_count == len(gs):
        check_dtype = False

    assert_eq(ps.dropna(), gs.dropna(), check_dtype=check_dtype)
