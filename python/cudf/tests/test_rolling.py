import pandas as pd
import numpy as np

import cudf
from cudf.tests.utils import assert_eq


def test_rollling_series_basic():
    psr = pd.Series([1, 2, 9, 4])
    gsr = cudf.Series([1, 2, 9, 4])
    assert_eq(psr.rolling(2).sum().fillna(-1),
              gsr.rolling(2).sum().fillna(-1),
              check_dtype=False)
