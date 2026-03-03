# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import rmm

import pylibcudf as plc
from pylibcudf.null_mask import MaskState


@pytest.fixture(params=[False, True])
def nullable(request):
    return request.param


@pytest.fixture(params=["float32", "float64"])
def column(request, nullable):
    values = [2.5, 2.49, 1.6, 8, -1.5, -1.7, -0.5, 0.5]
    typ = {"float32": pa.float32(), "float64": pa.float64()}[request.param]
    if nullable:
        values[2] = None
    return plc.Column.from_arrow(pa.array(values, type=typ))


def test_copy_bitmask(column, nullable):
    expected = (
        column.null_mask()
        if nullable
        else plc.gpumemoryview(rmm.DeviceBuffer())
    )
    got = plc.gpumemoryview(plc.null_mask.copy_bitmask(column))

    start = 0
    end = column.size() * 8
    assert plc.null_mask.null_count(
        expected, start, end
    ) == plc.null_mask.null_count(got, start, end)


def test_bitmask_allocation_size_bytes():
    assert plc.null_mask.bitmask_allocation_size_bytes(0) == 0
    assert plc.null_mask.bitmask_allocation_size_bytes(1) == 64
    assert plc.null_mask.bitmask_allocation_size_bytes(512) == 64
    assert plc.null_mask.bitmask_allocation_size_bytes(513) == 128
    assert plc.null_mask.bitmask_allocation_size_bytes(1024) == 128
    assert plc.null_mask.bitmask_allocation_size_bytes(1025) == 192


@pytest.mark.parametrize("size", [0, 1, 512, 1024])
@pytest.mark.parametrize(
    "state",
    [
        MaskState.UNALLOCATED,
        MaskState.UNINITIALIZED,
        MaskState.ALL_VALID,
        MaskState.ALL_NULL,
    ],
)
def test_create_null_mask(size, state):
    mask = plc.null_mask.create_null_mask(size, state)

    assert mask.size == (
        0
        if state == MaskState.UNALLOCATED
        else plc.null_mask.bitmask_allocation_size_bytes(size)
    )


def test_copy_bitmask_from_bitmask_invalid_span():
    class Foo:
        pass

    invalid_bitmask = Foo()

    with pytest.raises(
        TypeError,
        match="bitmask must satisfy Span protocol \\(have \\.ptr and \\.size\\), got Foo",
    ):
        plc.null_mask.copy_bitmask_from_bitmask(invalid_bitmask, 0, 10)


def test_null_count_invalid_span():
    class Foo:
        pass

    invalid_bitmask = Foo()

    with pytest.raises(
        TypeError,
        match="bitmask must satisfy Span protocol \\(have \\.ptr and \\.size\\), got Foo",
    ):
        plc.null_mask.null_count(invalid_bitmask, 0, 10)
