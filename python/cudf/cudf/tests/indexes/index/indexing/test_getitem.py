# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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


@pytest.mark.parametrize("idx", [0, np.int64(0)])
def test_index_getitem_from_int(idx):
    result = cudf.Index([1, 2])[idx]
    assert result == 1


@pytest.mark.parametrize("idx", [1.5, True, "foo"])
def test_index_getitem_from_nonint_raises(idx):
    with pytest.raises(ValueError):
        cudf.Index([1, 2])[idx]
