# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc


@pytest.fixture(scope="module")
def string_col():
    return pa.array(
        ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]
    )


def wrap_nulls(func):
    def wrapper(x):
        if x is None:
            return None
        return func(x)

    return wrapper


def test_to_upper(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.to_upper(plc_col)

    @wrap_nulls
    def to_upper(x):
        return x.upper()

    expected = pa.Array.from_pandas(string_col.to_pandas().apply(to_upper))
    assert_column_eq(got, expected)


def test_to_lower(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.to_lower(plc_col)

    @wrap_nulls
    def to_lower(x):
        return x.lower()

    expected = pa.Array.from_pandas(string_col.to_pandas().apply(to_lower))
    assert_column_eq(got, expected)


def test_swapcase(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = plc.strings.case.swapcase(plc_col)

    @wrap_nulls
    def swapcase(x):
        return x.swapcase()

    expected = pa.Array.from_pandas(string_col.to_pandas().apply(swapcase))
    assert_column_eq(got, expected)
