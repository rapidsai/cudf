from numba import cuda
import numpy as np
import pytest
import pandas as pd

import cudf
from cudf.tests.utils import assert_eq

def test_dataframe_apply_rows_null_mask():

    def kernel_multiply(a, b, out):
        for i, (x, y) in enumerate(zip(a, b)):
            out[i] = x * y

    df_original = cudf.DataFrame({
        'a': [3.0, None, 9.0,  None, 4.0],
        'b': [3.0, 7.0,  None, None, 2.0],
    })

    df_expected = cudf.DataFrame({
        'a':   [3.0, None, 9.0,  None, 4.0],
        'b':   [3.0, 7.0,  None, None, 2.0],
        'out': [9.0, None, None, None, 8.0]
    })

    df_actual = df_original.apply_rows(kernel_multiply, ['a', 'b'], { 'out': float }, {})

    assert_eq(df_expected, df_actual)

