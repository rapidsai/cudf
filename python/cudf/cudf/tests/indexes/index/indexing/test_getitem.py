# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "index_values",
    [range(1, 10, 2), [1, 2, 3], ["a", "b", "c"], [1.5, 2.5, 3.5]],
)
@pytest.mark.parametrize("i_type", [int, np.int8, np.int32, np.int64])
def test_scalar_getitem(index_values, i_type):
    i = i_type(1)
    index = cudf.Index(index_values)

    assert not isinstance(index[i], cudf.Index)
    assert index[i] == index_values[i]
    assert_eq(index, index.to_pandas())
