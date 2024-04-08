# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import cudf._lib.pylibcudf as plc
from cudf._lib.pylibcudf.strings import case


@pytest.fixture(scope="module")
def string_col():
    return pa.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])


def test_to_upper(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = case.to_upper(plc_col)
    expected = pa.Array.from_pandas(
        string_col.to_pandas().apply(lambda x: x.upper())
    )
    assert_column_eq(got, expected)


def test_to_lower(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = case.to_lower(plc_col)
    expected = pa.Array.from_pandas(
        string_col.to_pandas().apply(lambda x: x.lower())
    )
    assert_column_eq(got, expected)


def test_swapcase(string_col):
    plc_col = plc.interop.from_arrow(string_col)
    got = case.swapcase(plc_col)
    expected = pa.Array.from_pandas(
        string_col.to_pandas().apply(lambda x: x.swapcase())
    )
    assert_column_eq(got, expected)
