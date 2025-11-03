# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.ir import broadcast
from cudf_polars.utils.cuda_stream import get_cuda_stream


@pytest.mark.parametrize("target", [4, None])
def test_broadcast_all_scalar(target):
    stream = get_cuda_stream()
    columns = [
        Column(
            plc.column_factories.make_numeric_column(
                plc.DataType(plc.TypeId.INT8), 1, plc.MaskState.ALL_VALID, stream=stream
            ),
            name=f"col{i}",
            dtype=DataType(pl.Int8()),
        )
        for i in range(3)
    ]
    result = broadcast(*columns, target_length=target, stream=stream)
    expected = 1 if target is None else target

    assert [c.name for c in result] == [f"col{i}" for i in range(3)]
    assert all(column.size == expected for column in result)


def test_invalid_target_length():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    columns = [
        Column(
            plc.column_factories.make_numeric_column(
                dtype.plc_type, 4, plc.MaskState.ALL_VALID, stream=stream
            ),
            dtype=dtype,
            name=f"col{i}",
        )
        for i in range(3)
    ]
    with pytest.raises(RuntimeError):
        _ = broadcast(*columns, target_length=8, stream=stream)


def test_broadcast_mismatching_column_lengths():
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    columns = [
        Column(
            plc.column_factories.make_numeric_column(
                dtype.plc_type, i + 1, plc.MaskState.ALL_VALID, stream=stream
            ),
            dtype=dtype,
            name=f"col{i}",
        )
        for i in range(3)
    ]
    with pytest.raises(RuntimeError):
        _ = broadcast(*columns, stream=stream)


@pytest.mark.parametrize("nrows", [0, 5])
def test_broadcast_with_scalars(nrows):
    stream = get_cuda_stream()
    dtype = DataType(pl.Int8())
    columns = [
        Column(
            plc.column_factories.make_numeric_column(
                dtype.plc_type,
                nrows if i == 0 else 1,
                plc.MaskState.ALL_VALID,
                stream=stream,
            ),
            dtype=dtype,
            name=f"col{i}",
        )
        for i in range(3)
    ]

    result = broadcast(*columns, stream=stream)
    assert [c.name for c in result] == [f"col{i}" for i in range(3)]
    assert all(column.size == nrows for column in result)
