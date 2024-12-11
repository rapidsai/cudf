# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc


@pytest.fixture(scope="module")
def reshape_data():
    data = [[1, 2, 3], [4, 5, 6]]
    arrow_tbl = pa.Table.from_arrays(data, names=["a", "b"])
    return data, plc.interop.from_arrow(arrow_tbl)


def test_interleave_columns(reshape_data):
    raw_data, reshape_plc_tbl = reshape_data
    res = plc.reshape.interleave_columns(reshape_plc_tbl)

    interleaved_data = [pa.array(pair) for pair in zip(*raw_data)]

    expect = pa.concat_arrays(interleaved_data)

    assert_column_eq(expect, res)


@pytest.mark.parametrize("cnt", [0, 1, 3])
def test_tile(reshape_data, cnt):
    raw_data, reshape_plc_tbl = reshape_data
    res = plc.reshape.tile(reshape_plc_tbl, cnt)

    tiled_data = [pa.array(col * cnt) for col in raw_data]

    expect = pa.Table.from_arrays(
        tiled_data, schema=plc.interop.to_arrow(reshape_plc_tbl).schema
    )

    assert_table_eq(expect, res)
