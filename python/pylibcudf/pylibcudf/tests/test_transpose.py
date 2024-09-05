# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        {"a": [1, 2], "b": [1, 2]},
        {"a": [1], "b": [1]},
    ],
)
def test_transpose(data):
    arrow_tbl = pa.table(data)
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    plc.transpose.transpose(plc_tbl)
