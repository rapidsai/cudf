# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc

param_pyarrow_tables = [
    pa.table([]),
    pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
    pa.table({"a": [1, 2, 3]}),
    pa.table({"a": [1], "b": [2], "c": [3]}),
    pa.table({"a": ["a", "bb", "ccc"]}),
    pa.table({"a": [1, 2, None], "b": [None, 3, 4]}),
    pa.table(
        {
            "a": [["a", "b"], ["cde"]],
            "b": [
                {"alpha": [1, 2], "beta": None},
                {"alpha": [3, 4], "beta": 5},
            ],
        }
    ),
]


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_pack_and_unpack(arrow_tbl):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    packed = plc.contiguous_split.pack(plc_tbl)

    res = plc.contiguous_split.unpack(packed)
    assert_table_eq(arrow_tbl, res)


@pytest.mark.parametrize("arrow_tbl", param_pyarrow_tables)
def test_pack_and_unpack_from_memoryviews(arrow_tbl):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    packed = plc.contiguous_split.pack(plc_tbl)

    metadata, gpudata = packed.release()

    with pytest.raises(ValueError, match="Cannot release empty"):
        packed.release()

    del packed  # `metadata` and `gpudata` will survive

    res = plc.contiguous_split.unpack_from_memoryviews(metadata, gpudata)
    assert_table_eq(arrow_tbl, res)
