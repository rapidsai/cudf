# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_integers():
    typ = pa.int8()
    arr = pa.array(["1", "-1", None])
    got = plc.strings.convert.convert_integers.to_integers(
        plc.Column.from_arrow(arr), plc.DataType.from_arrow(typ)
    )
    expect = arr.cast(typ)
    assert_column_eq(expect, got)


def test_from_integers():
    arr = pa.array([1, -1, None])
    got = plc.strings.convert.convert_integers.from_integers(
        plc.Column.from_arrow(arr)
    )
    expect = pa.array(["1", "-1", None])
    assert_column_eq(expect, got)


def test_is_integer():
    arr = pa.array(["1", "-1", "1.2", "A", None])
    plc_column = plc.Column.from_arrow(arr)
    got = plc.strings.convert.convert_integers.is_integer(plc_column)
    expect = pa.array([True, True, False, False, None])
    assert_column_eq(expect, got)

    got = plc.strings.convert.convert_integers.is_integer(
        plc_column, plc.DataType.from_arrow(pa.uint8())
    )
    expect = pa.array([True, False, False, False, None])
    assert_column_eq(expect, got)


def test_hex_to_integers():
    typ = pa.int32()
    data = ["0xff", "0x2a", None]
    got = plc.strings.convert.convert_integers.hex_to_integers(
        plc.Column.from_arrow(pa.array(data)), plc.DataType.from_arrow(typ)
    )
    expect = pa.array(
        [int(val, 16) if isinstance(val, str) else val for val in data],
        type=typ,
    )
    assert_column_eq(expect, got)


def test_is_hex():
    arr = pa.array(["0xff", "123", "!", None])
    got = plc.strings.convert.convert_integers.is_hex(
        plc.Column.from_arrow(arr)
    )
    expect = pa.array([True, True, False, None])
    assert_column_eq(expect, got)


def test_integers_to_hex():
    data = [255, -42, None]
    arr = pa.array(data)
    got = plc.strings.convert.convert_integers.integers_to_hex(
        plc.Column.from_arrow(arr)
    )
    expect = pa.array(["FF", "FFFFFFFFFFFFFFD6", None])
    assert_column_eq(expect, got)
