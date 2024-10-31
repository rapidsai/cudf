# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
from utils import assert_column_eq

import pylibcudf as plc


def test_find_multiple():
    arr = pa.array(["abc", "def"])
    targets = pa.array(["a", "c", "e"])
    result = plc.strings.find_multiple.find_multiple(
        plc.interop.from_arrow(arr),
        plc.interop.from_arrow(targets),
    )
    expected = pa.array(
        [
            [elem.find(target) for target in targets.to_pylist()]
            for elem in arr.to_pylist()
        ],
        type=pa.list_(pa.int32()),
    )
    assert_column_eq(expected, result)
