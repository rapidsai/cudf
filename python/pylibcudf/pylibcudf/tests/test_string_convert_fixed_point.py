# Copyright (c) 2024, NVIDIA CORPORATION.
import decimal

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_fixed_point():
    typ = pa.decimal128(38, 2)
    arr = pa.array(["123", "1.23", None])
    result = plc.strings.convert.convert_fixed_point.to_fixed_point(
        plc.interop.from_arrow(arr), plc.interop.from_arrow(typ)
    )
    expected = arr.cast(typ)
    assert_column_eq(result, expected)


def test_from_fixed_point():
    arr = pa.array([decimal.Decimal("1.1"), None])
    result = plc.strings.convert.convert_fixed_point.from_fixed_point(
        plc.interop.from_arrow(arr),
    )
    expected = pa.array(["1.1", None])
    assert_column_eq(result, expected)


def test_is_fixed_point():
    arr = pa.array(["123", "1.23", "1.2.3", "", None])
    result = plc.strings.convert.convert_fixed_point.is_fixed_point(
        plc.interop.from_arrow(arr),
    )
    expected = pa.array([True, True, False, False, None])
    assert_column_eq(result, expected)
