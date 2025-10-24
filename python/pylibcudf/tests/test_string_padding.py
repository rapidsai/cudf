# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.compute as pc
from utils import assert_column_eq

import pylibcudf as plc


def test_pad():
    arr = pa.array(["a", "1", None])
    got = plc.strings.padding.pad(
        plc.Column.from_arrow(arr),
        2,
        plc.strings.side_type.SideType.LEFT,
        "!",
    )
    expect = pa.array(pc.utf8_lpad(arr, 2, padding="!"))
    assert_column_eq(expect, got)


def test_zfill():
    arr = pa.array(["a", "1", "-1", None])
    got = plc.strings.padding.zfill(plc.Column.from_arrow(arr), 2)
    expect = pa.array(["0a", "01", "-1", None])
    assert_column_eq(expect, got)


def test_zfill_by_widths():
    arr = pa.array(["a", "1", "-1", None])
    widths = pa.array([2, 3, 4, 0], type=pa.int32())
    got = plc.strings.padding.zfill_by_widths(
        plc.Column.from_arrow(arr), plc.Column.from_arrow(widths)
    )
    expect = pa.array(["0a", "001", "-001", None])
    assert_column_eq(expect, got)
