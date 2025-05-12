# Copyright (c) 2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def test_from_flat_list():
    data = [1, 2, 3]
    expected = pa.array(data, type=pa.int64())
    got = plc.Column.from_list(data)
    assert_column_eq(expected, got)


def test_from_list_of_lists():
    data = [[1, 2], [3, 4]]
    expected = pa.array(data, type=pa.list_(pa.int64()))
    got = plc.Column.from_list(data)
    assert_column_eq(expected, got)


def test_from_mixed_list_of_lists():
    with pytest.raises(ValueError, match="Inconsistent inner list shapes"):
        plc.Column.from_list([[1], [2, 3], [], [4]])
