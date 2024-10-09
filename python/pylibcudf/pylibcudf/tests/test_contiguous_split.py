# Copyright (c) 2024, NVIDIA CORPORATION.

import cupy
import pyarrow as pa
import pylibcudf as plc
import pytest
from utils import assert_table_eq


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
        pa.table({"a": ["a", "bb", "ccc"]}),
    ],
)
def test_pack_and_unpack(arrow_tbl):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    packed = plc.contiguous_split.pack(plc_tbl)

    res = plc.contiguous_split.unpack(packed)
    assert_table_eq(arrow_tbl, res)

    # Copy the buffers to simulate IO
    metadata = memoryview(bytes(packed.metadata))
    gpu_data = plc.gpumemoryview(cupy.array(packed.gpu_data, copy=True))

    res = plc.contiguous_split.unpack_from_memoryviews(metadata, gpu_data)
    assert_table_eq(arrow_tbl, res)
