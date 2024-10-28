# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import pylibcudf as plc


@pytest.mark.parametrize("repeats", [pa.array([2, 2]), 2])
def test_repeat_strings(repeats):
    arr = pa.array(["1", None])
    plc_result = plc.strings.repeat.repeat_strings(
        plc.interop.from_arrow(arr),
        plc.interop.from_arrow(repeats)
        if not isinstance(repeats, int)
        else repeats,
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(pc.binary_repeat(arr, repeats))
    assert result.equals(expected)
