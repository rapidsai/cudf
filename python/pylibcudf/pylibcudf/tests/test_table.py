# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

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
    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    plc_tbl_shape = (plc_tbl.num_rows(), plc_tbl.num_columns())
    assert plc_tbl_shape == arrow_tbl.shape
