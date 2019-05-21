
import pytest

import numpy as np
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq

def test_dropna():
    gs = cudf.Series([1, 2, None, 4])
    ps = pd.Series([1, 2, None, 4])
    assert_eq(ps.dropna(), gs.dropna())
