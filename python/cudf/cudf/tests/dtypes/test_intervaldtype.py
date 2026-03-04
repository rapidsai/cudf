# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType

import cudf
from cudf.testing import assert_eq


def test_from_pandas_intervaldtype():
    dtype = pd.IntervalDtype("int64", closed="left")
    result = cudf.from_pandas(dtype)
    expected = cudf.IntervalDtype("int64", closed="left")
    assert_eq(result, expected)


def test_intervaldtype_eq_string_with_attributes():
    dtype = cudf.IntervalDtype("int64", closed="left")
    assert dtype == "interval"
    assert dtype == "interval[int64, left]"


def test_empty_intervaldtype():
    # "older pandas" supported closed=None, cudf chooses not to support that
    pd_id = pd.IntervalDtype(closed="right")
    cudf_id = cudf.IntervalDtype()

    assert str(pd_id) == str(cudf_id)
    assert pd_id.subtype == cudf_id.subtype


def test_interval_dtype_pyarrow_round_trip(
    signed_integer_types_as_str, interval_closed
):
    pa_array = ArrowIntervalType(signed_integer_types_as_str, interval_closed)
    expect = pa_array
    got = cudf.IntervalDtype.from_arrow(expect).to_arrow()
    assert expect.equals(got)


def test_interval_dtype_from_pandas(
    signed_integer_types_as_str, interval_closed
):
    expect = cudf.IntervalDtype(
        signed_integer_types_as_str, closed=interval_closed
    )
    pd_type = pd.IntervalDtype(
        signed_integer_types_as_str, closed=interval_closed
    )
    got = cudf.IntervalDtype(pd_type.subtype, closed=pd_type.closed)
    assert expect == got
