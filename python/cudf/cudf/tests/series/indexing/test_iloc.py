# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
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


@pytest.mark.parametrize(
    "item",
    [
        0,
        2,
        4,
        slice(1, 3),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        np.array([0, 1, 2, 3, 4]),
        cp.asarray(np.array([0, 1, 2, 3, 4])),
    ],
)
@pytest.mark.parametrize("data", [["a"] * 5, ["a", None] * 3, [None] * 5])
def test_string_get_item(data, item):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    got = gs.iloc[item]
    if isinstance(got, cudf.Series):
        got = got.to_arrow()

    if isinstance(item, cp.ndarray):
        item = cp.asnumpy(item)

    expect = ps.iloc[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        if got is cudf.NA and expect is None:
            return
        assert expect == got


@pytest.mark.parametrize("bool_", [True, False])
@pytest.mark.parametrize("data", [["a"], ["a", None], [None]])
@pytest.mark.parametrize("box", [list, np.array, cp.array])
def test_string_bool_mask(data, bool_, box):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")
    item = box([bool_] * len(data))

    got = gs.iloc[item]
    if isinstance(got, cudf.Series):
        got = got.to_arrow()

    if isinstance(item, cp.ndarray):
        item = cp.asnumpy(item)

    expect = ps[item]
    if isinstance(expect, pd.Series):
        expect = pa.Array.from_pandas(expect)
        pa.Array.equals(expect, got)
    else:
        assert expect == got
