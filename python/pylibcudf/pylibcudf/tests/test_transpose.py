# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc


def test_transpose():
    arrow_tbl = pa.table({"a": [1, 2, 3]})
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    plc.transpose.transpose(plc_tbl)
