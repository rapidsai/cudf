# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
    got = plc.nvtext.edit_distance.edit_distance(
        plc.Column.from_arrow(input_col),
        plc.Column.from_arrow(targets),
    )
    expect = pa.array([1, 7, 0], type=pa.int32())
    assert_column_eq(expect, got)
