# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from cudf.tests.utils import NUMERIC_TYPES, OTHER_TYPES


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("exact", ["equiv", True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
# @pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_index_equal(
    rdata, exact, check_names, rname, check_categorical, dtype,
):
    p_left = pd.Index([1, 2, 3], name="a", dtype=dtype)
    p_right = pd.Index(rdata, name=rname, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_index_equal(
            p_left,
            p_right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )
    except BaseException as e:
        kind = type(e)
        msg = str(e)

    if kind is not None:
        if (kind == TypeError) and (
            msg
            == (
                "Categoricals can only be compared "
                "if 'categories' are the same."
            )
        ):
            kind = AssertionError
        with pytest.raises(kind):
            assert_index_equal(
                left,
                right,
                exact=exact,
                check_names=check_names,
                check_categorical=check_categorical,
            )
    else:
        assert_index_equal(
            left,
            right,
            exact=exact,
            check_names=check_names,
            check_categorical=check_categorical,
        )


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("check_index_type", ["equiv", True, False])
@pytest.mark.parametrize("check_series_type", [True, False])
@pytest.mark.parametrize("check_dtype", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_category_order", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_series_equal(
    rdata,
    rname,
    check_index_type,
    check_series_type,
    check_dtype,
    check_names,
    check_exact,
    check_category_order,
    check_categorical,
    dtype,
):

    p_left = pd.Series([1, 2, 3], name="a", dtype=dtype)
    p_right = pd.Series(rdata, name=rname, dtype=dtype)

    left = cudf.from_pandas(p_left)
    right = cudf.from_pandas(p_right)

    kind = None
    try:
        pd.testing.assert_series_equal(
            p_left,
            p_right,
            check_index_type=check_index_type,
            check_series_type=check_series_type,
            check_dtype=check_dtype,
            check_names=check_names,
            check_exact=check_exact,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                left,
                right,
                check_index_type=check_index_type,
                check_series_type=check_series_type,
                check_dtype=check_dtype,
                check_names=check_names,
                check_exact=check_exact,
                check_categorical=check_categorical,
                check_category_order=check_category_order,
            )
    else:
        assert_series_equal(
            left,
            right,
            check_index_type=check_index_type,
            check_series_type=check_series_type,
            check_dtype=check_dtype,
            check_names=check_names,
            check_exact=check_exact,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
        )


@pytest.mark.parametrize(
    "rdtype", [["int8", "int16", "int64"], ["int64", "int16", "int8"]]
)
@pytest.mark.parametrize("rname", [["a", "b", "c"], ["b", "c", "a"]])
@pytest.mark.parametrize("index", [[1, 2, 3], [3, 2, 1]])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("check_dtype", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
@pytest.mark.parametrize("check_like", [True, False])
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

    kind = None
    try:
        pd.testing.assert_frame_equal(
            p_left,
            p_right,
            check_exact=check_exact,
            check_dtype=check_dtype,
            check_names=check_names,
            check_like=check_like,
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_frame_equal(
                left,
                right,
                check_exact=check_exact,
                check_dtype=check_dtype,
                check_names=check_names,
                check_like=check_like,
            )
    else:
        assert_frame_equal(
            left,
            right,
            check_exact=check_exact,
            check_dtype=check_dtype,
            check_names=check_names,
            check_like=check_like,
        )
