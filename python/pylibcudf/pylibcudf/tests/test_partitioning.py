# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def partitioning_data():
    data = {"a": [1, 2, 3], "b": [1, 2, 5], "c": [1, 2, 10]}
    pa_table = pa.table(data)
    plc_table = plc.interop.from_arrow(pa_table)
    return data, plc_table, pa_table


def test_partition(partitioning_data):
    raw_data, plc_table, pa_table = partitioning_data
    result, result_offsets = plc.partitioning.partition(
        plc_table,
        plc.interop.from_arrow(pa.array([0, 0, 0])),
        1,
    )
    expected = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expected, result)
    assert result_offsets == [0, 3]


def test_hash_partition(partitioning_data):
    raw_data, plc_table, pa_table = partitioning_data
    result, result_offsets = plc.partitioning.hash_partition(
        plc_table, [0, 1], 1
    )
    expected = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expected, result)
    assert result_offsets == [0]


def test_round_robin_partition(partitioning_data):
    raw_data, plc_table, pa_table = partitioning_data
    result, result_offsets = plc.partitioning.round_robin_partition(
        plc_table, 1, 0
    )
    expected = pa.table(
        list(raw_data.values()),
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert_table_eq(expected, result)
    assert result_offsets == [0]
