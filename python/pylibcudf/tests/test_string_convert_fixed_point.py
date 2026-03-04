# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import decimal

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_fixed_point():
    typ = pa.decimal128(38, 2)
    arr = pa.array(["123", "1.23", None])
    got = plc.strings.convert.convert_fixed_point.to_fixed_point(
        plc.Column.from_arrow(arr), plc.DataType.from_arrow(typ)
    )
    expect = arr.cast(typ)
    assert_column_eq(expect, got)


def test_from_fixed_point():
    arr = pa.array([decimal.Decimal("1.1"), None])
    got = plc.strings.convert.convert_fixed_point.from_fixed_point(
        plc.Column.from_arrow(arr),
    )
    expect = pa.array(["1.1", None])
    assert_column_eq(expect, got)


def test_is_fixed_point():
    arr = pa.array(["123", "1.23", "1.2.3", "", None])
    got = plc.strings.convert.convert_fixed_point.is_fixed_point(
        plc.Column.from_arrow(arr),
    )
    expect = pa.array([True, True, False, False, None])
    assert_column_eq(expect, got)
