# Copyright (c) 2024-2025, NVIDIA CORPORATION.

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
    arr = pa.array(["a", "1", None])
    got = plc.strings.padding.zfill(plc.Column.from_arrow(arr), 2)
    expect = pa.array(pc.utf8_lpad(arr, 2, padding="0"))
    assert_column_eq(expect, got)
