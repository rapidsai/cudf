# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import operator
import re

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_series_where_mixed_dtypes_error():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=re.escape(
            "cudf does not support mixed types, please type-cast "
            "the column of dataframe/series and other "
            "to same dtypes."
        ),
    ):
        s.where([True, False, True], [1, 2, 3])


def test_series_where_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s.where(~s, 10)


@pytest.mark.parametrize("fill_value", [100, 100.0, 128.5])
@pytest.mark.parametrize("op", [operator.gt, operator.eq, operator.lt])
def test_series_where(numeric_types_as_str, fill_value, op):
    psr = pd.Series(list(range(10)), dtype=numeric_types_as_str)
    sr = cudf.from_pandas(psr)

    try:
        scalar_fits = sr.dtype.type(fill_value) == fill_value
    except OverflowError:
        scalar_fits = False

    if not scalar_fits:
        with pytest.raises(TypeError):
            sr.where(op(sr, 0), fill_value)
    else:
        # Cast back to original dtype as pandas automatically upcasts
        expect = psr.where(op(psr, 0), fill_value)
        got = sr.where(op(sr, 0), fill_value)
        # pandas returns 'float16' dtype, which is not supported in cudf
        assert_eq(
            expect,
            got,
            check_dtype=expect.dtype.kind != "f",
        )


@pytest.mark.parametrize("fill_value", [100, 100.0, 100.5])
def test_series_with_nulls_where(fill_value):
    psr = pd.Series([None] * 3 + list(range(5)))
    sr = cudf.from_pandas(psr)

    expect = psr.where(psr > 0, fill_value)
    got = sr.where(sr > 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr < 0, fill_value)
    got = sr.where(sr < 0, fill_value)
    assert_eq(expect, got)

    expect = psr.where(psr == 0, fill_value)
    got = sr.where(sr == 0, fill_value)
    assert_eq(expect, got)
