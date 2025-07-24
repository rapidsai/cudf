# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_frame_equal
from cudf.testing._utils import assert_asserters_equal


@pytest.fixture(params=[True, False])
def check_like(request):
    """Argument for assert_frame_equal"""
    return request.param


@pytest.mark.parametrize(
    "rdtype", [["int8", "int16", "int64"], ["int64", "int16", "int8"]]
)
@pytest.mark.parametrize("rname", [["a", "b", "c"], ["b", "c", "a"]])
@pytest.mark.parametrize("index", [[1, 2, 3], [3, 2, 1]])
@pytest.mark.parametrize("mismatch", [True, False])
def test_basic_assert_frame_equal(
    rdtype,
    rname,
    index,
    check_exact,
    check_dtype,
    check_names,
    check_like,
    mismatch,
):
    data = [1, 2, 1]
    p_left = pd.DataFrame(index=[1, 2, 3])
    p_left["a"] = np.array(data, dtype="int8")
    p_left["b"] = np.array(data, dtype="int16")
    if mismatch:
        p_left["c"] = np.array([1, 2, 3], dtype="int64")
    else:
        p_left["c"] = np.array(data, dtype="int64")

    p_right = pd.DataFrame(index=index)
    for dtype, name in zip(rdtype, rname):
        p_right[name] = np.array(data, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    assert_asserters_equal(
        pd.testing.assert_frame_equal,
        assert_frame_equal,
        p_left,
        p_right,
        left,
        right,
        check_exact=check_exact,
        check_dtype=check_dtype,
        check_names=check_names,
        check_like=check_like,
    )
