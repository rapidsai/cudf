# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_index_equal
from cudf.testing._utils import assert_asserters_equal


@pytest.fixture(params=["equiv", True, False])
def exact(request):
    """Argument for assert_index_equal"""
    return request.param


def test_range_index_and_int_index_equality(
    signed_integer_types_as_str, exact
):
    pidx1 = pd.RangeIndex(0, stop=5, step=1)
    pidx2 = pd.Index([0, 1, 2, 3, 4])
    idx1 = cudf.from_pandas(pidx1)
    idx2 = cudf.Index([0, 1, 2, 3, 4], dtype=signed_integer_types_as_str)

    assert_asserters_equal(
        pd.testing.assert_index_equal,
        assert_index_equal,
        pidx1,
        pidx2,
        idx1,
        idx2,
        exact=exact,
    )


@pytest.mark.parametrize("rdata", [3, 4], ids=["same", "different"])
def test_multiindex_equal(rdata):
    pidx1 = pd.MultiIndex.from_arrays(
        [[0, 1, 2, 3], ["G", "O", "N", "E"]], names=("n", "id")
    )
    pidx2 = pd.MultiIndex.from_arrays(
        [[0, 1, 2, rdata], ["G", "O", "N", "E"]], names=("n", "id")
    )

    idx1 = cudf.from_pandas(pidx1)
    idx2 = cudf.from_pandas(pidx2)

    assert_asserters_equal(
        pd.testing.assert_index_equal,
        assert_index_equal,
        pidx1,
        pidx2,
        idx1,
        idx2,
    )


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_categorical", [True, False])
def test_basic_assert_index_equal(
    rdata,
    exact,
    check_names,
    rname,
    check_categorical,
    all_supported_types_as_str,
):
    p_left = pd.Index([1, 2, 3], name="a", dtype=all_supported_types_as_str)
    p_right = pd.Index(rdata, name=rname, dtype=all_supported_types_as_str)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    assert_asserters_equal(
        pd.testing.assert_index_equal,
        assert_index_equal,
        p_left,
        p_right,
        left,
        right,
        exact=exact,
        check_names=check_names,
        check_categorical=check_categorical,
    )
