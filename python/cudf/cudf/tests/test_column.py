# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq

dtypes = [
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "datetime64[ns]",
    "str",
    "category",
]


@pytest.fixture(params=dtypes, ids=dtypes)
def pandas_input(request):
    data = np.random.randint(0, 1000, 100)
    return pd.Series(data, dtype=request.param)


@pytest.mark.parametrize("offset", [None, 0, 1, 15])
@pytest.mark.parametrize("size", [None, 50, 10, 0])
def test_column_offset_and_size(pandas_input, offset, size):
    col = cudf.core.column.as_column(pandas_input)
    if offset is not None:
        if offset > col.size:
            with pytest.raises(RuntimeError):
                col.offset = offset
        else:
            col.offset = offset
    if size is not None:
        col.size = size

    got = cudf.Series(col)

    if offset is None:
        offset = 0
    if size is None:
        size = 100
    else:
        size = size + offset

    slicer = slice(offset, size)
    expect = pandas_input.iloc[slicer].reset_index(drop=True)

    assert_eq(expect, got)
