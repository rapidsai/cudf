# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_rangeindex_arg_validation():
    with pytest.raises(TypeError):
        cudf.RangeIndex("1")

    with pytest.raises(TypeError):
        cudf.RangeIndex(1, "2")

    with pytest.raises(TypeError):
        cudf.RangeIndex(1, 3, "1")

    with pytest.raises(ValueError):
        cudf.RangeIndex(1, dtype="float64")

    with pytest.raises(ValueError):
        cudf.RangeIndex(1, dtype="uint64")


def test_rangeindex_name_not_hashable():
    with pytest.raises(ValueError):
        cudf.RangeIndex(range(2), name=["foo"])

    with pytest.raises(ValueError):
        cudf.RangeIndex(range(2)).copy(name=["foo"])


@pytest.mark.parametrize("klass", [cudf.RangeIndex, pd.RangeIndex])
@pytest.mark.parametrize("name_inner", [None, "a"])
@pytest.mark.parametrize("name_outer", [None, "b"])
def test_rangeindex_accepts_rangeindex(klass, name_inner, name_outer):
    result = cudf.RangeIndex(klass(range(1), name=name_inner), name=name_outer)
    expected = pd.RangeIndex(
        pd.RangeIndex(range(1), name=name_inner), name=name_outer
    )
    assert_eq(result, expected)


def test_from_pandas_rangeindex_step():
    expected = pd.RangeIndex(start=0, stop=8, step=2, name="myindex")
    actual = cudf.from_pandas(expected)

    assert_eq(expected, actual)
