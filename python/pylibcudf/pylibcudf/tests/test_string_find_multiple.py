# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc


def test_find_multiple():
    arr = pa.array(["abc", "def"])
    targets = pa.array(["a", "c", "e"])
    plc_result = plc.strings.find_multiple.find_multiple(
        plc.interop.from_arrow(arr),
        plc.interop.from_arrow(targets),
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(
        [
            pa.array(
                [
                    [elem.find(target) for target in targets.to_pylist()]
                    for elem in arr.to_pylist()
                ],
                type=result.type,
            )
        ]
    )
    assert result.equals(expected)
