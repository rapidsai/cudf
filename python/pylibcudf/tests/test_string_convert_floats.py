# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_to_floats():
    typ = pa.float32()
    arr = pa.array(["-1.23", "1", None])
    got = plc.strings.convert.convert_floats.to_floats(
        plc.Column.from_arrow(arr), plc.DataType.from_arrow(typ)
    )
    expect = arr.cast(typ)
    assert_column_eq(expect, got)


def test_from_floats():
    arr = pa.array([-1.23, 1, None])
    got = plc.strings.convert.convert_floats.from_floats(
        plc.Column.from_arrow(arr),
    )
    expect = pa.array(["-1.23", "1.0", None])
    assert_column_eq(expect, got)


def test_is_float():
    arr = pa.array(["-1.23", "1", "1.2.3", "A", None])
    got = plc.strings.convert.convert_floats.is_float(
        plc.Column.from_arrow(arr),
    )
    expect = pa.array([True, True, False, False, None])
    assert_column_eq(expect, got)
