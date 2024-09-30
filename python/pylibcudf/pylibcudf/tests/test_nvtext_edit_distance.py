# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
from utils import assert_column_eq


def test_edit_distance():
    input_arr = pa.array(["hallo", "goodbye", "world"])
    targets = pa.array(["hello", "", "world"])
    result = plc.nvtext.edit_distance.edit_distance(
        plc.interop.from_arrow(input_arr),
        plc.interop.from_arrow(targets),
    )
    expected = pa.array([1, 7, 0], type=pa.int32())
    assert_column_eq(result, expected)


def test_edit_distance_matrix():
    input_arr = pa.array(["hallo", "goodbye", "world"])
    result = plc.nvtext.edit_distance.edit_distance_matrix(
        plc.interop.from_arrow(input_arr),
    )
    expected = pa.array(
        [[0, 7, 4], [7, 0, 6], [4, 6, 0]], type=pa.list_(pa.int32())
    )
    assert_column_eq(expected, result)
