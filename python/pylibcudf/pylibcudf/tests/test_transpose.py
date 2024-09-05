# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


@pytest.mark.parametrize(
    "arr",
    [
        [],
        [1, 2, 3],
        [1, 2],
        [1],
    ],
)
def test_transpose(arr):
    data = {"a": arr, "b": arr}
    arrow_tbl = pa.table(data)
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    _, plc_result = plc.transpose.transpose(plc_tbl)
    arrow_result = plc.interop.to_arrow(plc_result)
    if len(arr) == 0:
        expected = (0, 0)
    else:
        expected = (len(data), len(arr))
    assert arrow_result.shape == expected
