# Copyright (c) 2024, NVIDIA CORPORATION.
import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_integers():
    typ = pa.int8()
    arr = pa.array(["1", "-1", None])
    result = plc.strings.convert.convert_integers.to_integers(
        plc.interop.from_arrow(arr), plc.interop.from_arrow(typ)
    )
    expected = arr.cast(typ)
    assert_column_eq(result, expected)


def test_from_integers():
    arr = pa.array([1, -1, None])
    result = plc.strings.convert.convert_integers.from_integers(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array(["1", "-1", None])
    assert_column_eq(result, expected)


def test_is_integer():
    arr = pa.array(["1", "-1", "1.2", "A", None])
    plc_column = plc.interop.from_arrow(arr)
    result = plc.strings.convert.convert_integers.is_integer(plc_column)
    expected = pa.array([True, True, False, False, None])
    assert_column_eq(result, expected)

    result = plc.strings.convert.convert_integers.is_integer(
        plc_column, plc.interop.from_arrow(pa.uint8())
    )
    expected = pa.array([True, False, False, False, None])
    assert_column_eq(result, expected)


def test_hex_to_integers():
    typ = pa.int32()
    data = ["0xff", "0x2a", None]
    result = plc.strings.convert.convert_integers.hex_to_integers(
        plc.interop.from_arrow(pa.array(data)), plc.interop.from_arrow(typ)
    )
    expected = pa.array(
        [int(val, 16) if isinstance(val, str) else val for val in data],
        type=typ,
    )
    assert_column_eq(result, expected)


def test_is_hex():
    arr = pa.array(["0xff", "123", "!", None])
    result = plc.strings.convert.convert_integers.is_hex(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array([True, True, False, None])
    assert_column_eq(result, expected)


def test_integers_to_hex():
    data = [255, -42, None]
    arr = pa.array(data)
    result = plc.strings.convert.convert_integers.integers_to_hex(
        plc.interop.from_arrow(arr)
    )
    expected = pa.array(["FF", "FFFFFFFFFFFFFFD6", None])
    assert_column_eq(result, expected)
