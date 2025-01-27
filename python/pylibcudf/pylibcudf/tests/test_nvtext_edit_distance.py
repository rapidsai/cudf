# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def edit_distance_data():
    arr1 = ["hallo", "goodbye", "world"]
    arr2 = ["hello", "", "world"]
    return pa.array(arr1), pa.array(arr2)


def test_edit_distance(edit_distance_data):
    input_col, targets = edit_distance_data
    result = plc.nvtext.edit_distance.edit_distance(
        plc.interop.from_arrow(input_col),
        plc.interop.from_arrow(targets),
    )
    expected = pa.array([1, 7, 0], type=pa.int32())
    assert_column_eq(result, expected)


def test_edit_distance_matrix(edit_distance_data):
    input_col, _ = edit_distance_data
    result = plc.nvtext.edit_distance.edit_distance_matrix(
        plc.interop.from_arrow(input_col)
    )
    expected = pa.array(
        [[0, 7, 4], [7, 0, 6], [4, 6, 0]], type=pa.list_(pa.int32())
    )
    assert_column_eq(expected, result)
