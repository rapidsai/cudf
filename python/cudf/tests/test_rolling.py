import pandas as pd
import numpy as np
import pytest

import cudf
from cudf.tests.utils import assert_eq

@pytest.mark.parametrize(
    'agg',
    ['sum', 'min', 'max', 'mean', 'count']
)
def test_rollling_series_basic():
    psr = pd.Series([1, 2, 4, 9, 9, 4])
    gsr = cudf.Series([1, 2, 4, 9, 9, 4])
    assert_eq(getattr(psr.rolling(2), agg)().fillna(-1),
              getattr(gsr.rolling(2), agg)().fillna(-1),
              check_dtype=False)
