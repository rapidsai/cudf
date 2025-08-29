# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_setitem_singleton_range():
    sr = cudf.Series([1, 2, 3], dtype=np.int64)
    psr = sr.to_pandas()
    value = np.asarray([7], dtype=np.int64)
    sr.iloc[:1] = value
    psr.iloc[:1] = value
    assert_eq(sr, cudf.Series([7, 2, 3], dtype=np.int64))
    assert_eq(sr, psr, check_dtype=True)


@pytest.mark.parametrize(
    "indices",
    [slice(0, 3), slice(1, 4), slice(None, None, 2), slice(1, None, 2)],
    ids=[":3", "1:4", "0::2", "1::2"],
)
@pytest.mark.parametrize(
    "values",
    [[None, {}, {}, None], [{}, {}, {}, {}]],
    ids=["nulls", "no_nulls"],
)
def test_struct_empty_children_slice(indices, values):
    s = cudf.Series(values)
    actual = s.iloc[indices]
    expect = cudf.Series(values[indices], index=range(len(values))[indices])
    assert_eq(actual, expect)
