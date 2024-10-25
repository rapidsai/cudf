# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from packaging.version import parse

import pylibcudf as plc


@pytest.mark.skipif(
    parse(pa.__version__) < parse("16.0.0"),
    reason="https://github.com/apache/arrow/pull/40070",
)
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
    plc_result = plc.transpose.transpose(plc_tbl)
    result = plc.interop.to_arrow(plc_result)
    expected = pa.Table.from_pandas(
        arrow_tbl.to_pandas().T, preserve_index=False
    ).rename_columns([""] * len(arr))
    expected = pa.table(expected, schema=result.schema)
    assert result.equals(expected)
