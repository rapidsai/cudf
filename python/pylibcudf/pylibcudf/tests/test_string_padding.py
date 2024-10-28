# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc

import pylibcudf as plc


def test_pad():
    arr = pa.array(["a", "1", None])
    plc_result = plc.strings.padding.pad(
        plc.interop.from_arrow(arr),
        2,
        plc.strings.side_type.SideType.LEFT,
        "!",
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(pc.utf8_lpad(arr, 2, padding="!"))
    assert result.equals(expected)


def test_zfill():
    arr = pa.array(["a", "1", None])
    plc_result = plc.strings.padding.zfill(plc.interop.from_arrow(arr), 2)
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(pc.utf8_lpad(arr, 2, padding="0"))
    assert result.equals(expected)
