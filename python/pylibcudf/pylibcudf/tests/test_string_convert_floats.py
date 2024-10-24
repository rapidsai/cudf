# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_floats():
    typ = pa.float32()
    arr = pa.array(["-1.23", "1", None])
    result = plc.strings.convert.convert_floats.to_floats(
        plc.interop.from_arrow(arr), plc.interop.from_arrow(typ)
    )
    expected = arr.cast(typ)
    assert_column_eq(result, expected)


def test_from_floats():
    arr = pa.array([-1.23, 1, None])
    result = plc.strings.convert.convert_floats.from_floats(
        plc.interop.from_arrow(arr),
    )
    expected = pa.array(["-1.23", "1.0", None])
    assert_column_eq(result, expected)


def test_is_float():
    arr = pa.array(["-1.23", "1", "1.2.3", "A", None])
    result = plc.strings.convert.convert_floats.is_float(
        plc.interop.from_arrow(arr),
    )
    expected = pa.array([True, True, False, False, None])
    assert_column_eq(result, expected)
