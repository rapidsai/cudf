# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest

import rmm

import cudf._lib.pylibcudf as plc


@pytest.fixture(params=[False, True])
def nullable(request):
    return request.param


@pytest.fixture(params=["float32", "float64"])
def column(request, nullable):
    values = [2.5, 2.49, 1.6, 8, -1.5, -1.7, -0.5, 0.5]
    typ = {"float32": pa.float32(), "float64": pa.float64()}[request.param]
    if nullable:
        values[2] = None
    return plc.interop.from_arrow(pa.array(values, type=typ))


def test_copy_bitmask(column, nullable):
    expected = column.null_mask().obj if nullable else rmm.DeviceBuffer()
    got = plc.null_mask.copy_bitmask(column)

    assert expected.size == got.size
    assert expected.tobytes() == got.tobytes()


def test_bitmask_allocation_size_bytes():
    assert plc.null_mask.bitmask_allocation_size_bytes(0) == 0
    assert plc.null_mask.bitmask_allocation_size_bytes(1) == 64
    assert plc.null_mask.bitmask_allocation_size_bytes(512) == 64
    assert plc.null_mask.bitmask_allocation_size_bytes(513) == 128
    assert plc.null_mask.bitmask_allocation_size_bytes(1024) == 128
    assert plc.null_mask.bitmask_allocation_size_bytes(1025) == 192
