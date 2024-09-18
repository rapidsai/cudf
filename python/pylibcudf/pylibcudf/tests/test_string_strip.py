# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc


def test_strip():
    arr = pa.array([" abc", " 123  ", None])
    plc_result = plc.strings.strip.strip(
        plc.interop.from_arrow(arr),
        plc.strings.side_type.SideType.BOTH,
        plc.interop.from_arrow(pa.scalar(" ")),
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(pc.utf8_trim_whitespace(arr))
    assert result.equals(expected)
