# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


@pytest.fixture
def pa_table():
    return pa.table({"a": [1, 2, 3], "b": [1, 2, 5], "c": [1, 2, 10]})


def test_partition(pa_table):
    plc_result, result_offsets = plc.partitioning.partition(
        plc.interop.from_arrow(pa_table),
        plc.interop.from_arrow(pa.array([0, 0, 0])),
        1,
    )
    pa_result = plc.interop.to_arrow(plc_result)
    pa_expected = pa.table(
        [[1, 2, 3], [1, 2, 5], [1, 2, 10]],
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert pa_result.equals(pa_expected)
    assert result_offsets == [0, 3]


def test_hash_partition(pa_table):
    plc_result, result_offsets = plc.partitioning.hash_partition(
        plc.interop.from_arrow(pa_table), [0, 1], 1
    )
    pa_result = plc.interop.to_arrow(plc_result)
    pa_expected = pa.table(
        [[1, 2, 3], [1, 2, 5], [1, 2, 10]],
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert pa_result.equals(pa_expected)
    assert result_offsets == [0]


def test_round_robin_partition(pa_table):
    plc_result, result_offsets = plc.partitioning.round_robin_partition(
        plc.interop.from_arrow(pa_table), 1, 0
    )
    pa_result = plc.interop.to_arrow(plc_result)
    pa_expected = pa.table(
        [[1, 2, 3], [1, 2, 5], [1, 2, 10]],
        schema=pa.schema([pa.field("", pa.int64(), nullable=False)] * 3),
    )
    assert pa_result.equals(pa_expected)
    assert result_offsets == [0]
