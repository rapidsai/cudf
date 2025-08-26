# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
    ],
)
def test_table_shape(arrow_tbl):
    plc_tbl = plc.Table.from_arrow(arrow_tbl)

    assert plc_tbl.shape() == arrow_tbl.shape


def test_table_to_arrow(table_data):
    plc_tbl, _ = table_data
    expect = plc_tbl.tbl
    got = expect.to_arrow()
    # Normally the order is expect, got
    # but assert_table_eq takes the pyarrow
    # table as it's first argument
    assert_table_eq(got, expect)
