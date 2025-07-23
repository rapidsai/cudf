# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.testing import (
    assert_series_equal,
)
from cudf.testing._utils import (
    NUMERIC_TYPES,
    OTHER_TYPES,
)


def test_series_different_type_cases(
    numeric_types_as_str, check_exact, check_dtype
):
    data = [0, 1, 2, 3]

    psr1 = pd.Series(data, dtype="uint8")
    psr2 = pd.Series(data, dtype=numeric_types_as_str)

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)

    kind = None
    try:
        pd.testing.assert_series_equal(
            psr1, psr2, check_exact=check_exact, check_dtype=check_dtype
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                sr1, sr2, check_exact=check_exact, check_dtype=check_dtype
            )
    else:
        assert_series_equal(
            sr1, sr2, check_exact=check_exact, check_dtype=check_dtype
        )


@pytest.mark.parametrize("rdata", [3, 4], ids=["same", "different"])
def test_datetime_like_compaibility(rdata, check_datetimelike_compat):
    psr1 = pd.Series([0, 1, 2, 3], dtype="datetime64[ns]")
    psr2 = pd.Series([0, 1, 2, rdata], dtype="datetime64[ns]").astype("str")

    sr1 = cudf.from_pandas(psr1)
    sr2 = cudf.from_pandas(psr2)

    kind = None
    try:
        pd.testing.assert_series_equal(
            psr1, psr2, check_datetimelike_compat=check_datetimelike_compat
        )
    except BaseException as e:
        kind = type(e)

    if kind is not None:
        with pytest.raises(kind):
            assert_series_equal(
                sr1, sr2, check_datetimelike_compat=check_datetimelike_compat
            )
    else:
        assert_series_equal(
            sr1, sr2, check_datetimelike_compat=check_datetimelike_compat
        )


@pytest.mark.parametrize("rdata", [[1, 2, 5], [1, 2, 6], [1, 2, 5, 6]])
@pytest.mark.parametrize("rname", ["a", "b"])
@pytest.mark.parametrize("check_category_order", [True, False])
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + OTHER_TYPES + ["datetime64[ns]"]
)
def test_basic_assert_series_equal(
    rdata,
    rname,
    check_names,
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
            check_names=check_names,
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
                check_names=check_names,
                check_categorical=check_categorical,
                check_category_order=check_category_order,
            )
    else:
        assert_series_equal(
            left,
            right,
            check_names=check_names,
            check_categorical=check_categorical,
            check_category_order=check_category_order,
        )
