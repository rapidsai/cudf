from numba import cuda
import numpy as np
import pytest
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq


def _kernel_multiply(a, b, out):
    for i, (x, y) in enumerate(zip(a, b)):
        out[i] = x * y


def _test_multiply(a, b, out):
    df_original = cudf.DataFrame({ 'a': a, 'b': b })
    df_expected = cudf.DataFrame({ 'a': a, 'b': b, 'out': out })
    df_actual = df_original.apply_rows(_kernel_multiply, ['a', 'b'], { 'out': float }, {})
    assert_eq(df_expected, df_actual)


def test_dataframe_apply_rows_null_mask():
    _test_multiply(
        [3.0, None, 9.0,  None, 4.0],
        [3.0, 7.0,  None, None, 2.0],
        [9.0, None, None, None, 8.0]
    )


def test_dataframe_apply_rows_null_mask_including_nan():
    _test_multiply(
        [3.0, None, 9.0,  None, np.nan, 5.0,    np.nan, 4.0],
        [3.0, 7.0,  None, None, 9.0,    np.nan, np.nan, 2.0],
        [9.0, None, None, None, np.nan, np.nan, np.nan, 8.0]
    )


def test_dataframe_apply_rows_empty():
    _test_multiply(
        [],
        [],
        []
    )


def test_dataframe_apply_rows_only_none():
    _test_multiply(
        [None],
        [None],
        [None]
    )
