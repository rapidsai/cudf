# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

from cudf._lib import pylibcudf as plc


def test_interleave_columns():
    data = [[1, 2, 3], [4, 5, 6]]
    arrow_tbl = pa.Table.from_arrays(data, names=["a", "b"])

    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    res = plc.reshape.interleave_columns(plc_tbl)

    interleaved_data = [pa.array(pair) for pair in zip(*data)]

    expect = pa.concat_arrays(interleaved_data)

    assert_column_eq(res, expect)


@pytest.mark.parametrize("cnt", [0, 1, 3])
def test_tile(cnt):
    data = [[1, 2, 3], [4, 5, 6]]
    arrow_tbl = pa.Table.from_arrays(data, names=["a", "b"])

    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    res = plc.reshape.tile(plc_tbl, cnt)

    tiled_data = [pa.array(col * cnt) for col in data]

    expect = pa.Table.from_arrays(tiled_data, schema=arrow_tbl.schema)

    assert_table_eq(res, expect)
