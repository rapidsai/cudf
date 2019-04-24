from __future__ import division, print_function

import pytest
import random
import numpy as np

from itertools import product
from cudf.dataframe import Series, column, columnops
from cudf.tests import utils
from cudf.tests.utils import gen_rand

import cudf.bindings.copying as cpp_copying


def test_gather_single_col():
    col = columnops.as_column(np.arange(100))
    gather_map = [0, 1, 2, 3, 5, 8, 13, 21]

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather_column(col, device_gather_map)

    assert out.to_pandas() == gather_map


def test_gather_cols():
    cols = [columnops.as_column(np.arange(10)), columnops.as_column(np.arange(10))]
    gather_map = [0, 1, 2, 3, 5, 8, 13, 21]

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather(cols, device_gather_map)

    assert out.to_pandas() == gather_map
