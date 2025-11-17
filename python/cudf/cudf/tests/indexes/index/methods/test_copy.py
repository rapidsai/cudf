# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_column_memory_eq,
    assert_column_memory_ne,
)


@pytest.mark.parametrize(
    "data",
    [
        range(1, 5),
        [1, 2, 3, 4],
        pd.DatetimeIndex(["2001", "2002", "2003"]),
        ["a", "b", "c"],
        pd.CategoricalIndex(["a", "b", "c"]),
    ],
)
@pytest.mark.parametrize("copy_on_write", [True, False])
def test_index_copy(data, deep, copy_on_write):
    name = "x"
    cidx = cudf.Index(data)
    pidx = cidx.to_pandas()

    pidx_copy = pidx.copy(name=name, deep=deep)
    cidx_copy = cidx.copy(name=name, deep=deep)

    assert_eq(pidx_copy, cidx_copy)

    with cudf.option_context("copy_on_write", copy_on_write):
        if not isinstance(cidx, cudf.RangeIndex):
            if not deep:
                # When `copy_on_write` is turned on, Index objects will
                # have unique column object but they all point to same
                # data pointers.
                assert_column_memory_eq(cidx._column, cidx_copy._column)
            else:
                assert_column_memory_ne(cidx._column, cidx_copy._column)
