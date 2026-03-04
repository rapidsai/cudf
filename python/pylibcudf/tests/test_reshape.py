# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pyarrow as pa
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc
from pylibcudf.types import TypeId


@pytest.fixture(scope="module")
def reshape_data():
    data = [[1, 2, 3], [4, 5, 6]]
    arrow_tbl = pa.Table.from_arrays(data, names=["a", "b"])
    return data, plc.Table.from_arrow(arrow_tbl)


def test_interleave_columns(reshape_data):
    raw_data, reshape_plc_tbl = reshape_data
    got = plc.reshape.interleave_columns(reshape_plc_tbl)

    interleaved_data = [pa.array(pair) for pair in zip(*raw_data, strict=True)]

    expect = pa.concat_arrays(interleaved_data)

    assert_column_eq(expect, got)


@pytest.mark.parametrize("cnt", [0, 1, 3])
def test_tile(reshape_data, cnt):
    raw_data, reshape_plc_tbl = reshape_data
    got = plc.reshape.tile(reshape_plc_tbl, cnt)

    expect = pa.Table.from_arrays(
        [pa.array(col * cnt) for col in raw_data],
        schema=reshape_plc_tbl.to_arrow().schema,
    )

    assert_table_eq(expect, got)


@pytest.mark.parametrize(
    "dtype, type_id",
    [
        ("int32", TypeId.INT32),
        ("int64", TypeId.INT64),
        ("float32", TypeId.FLOAT32),
        ("float64", TypeId.FLOAT64),
    ],
)
def test_table_to_array(dtype, type_id):
    arrow_type = pa.from_numpy_dtype(getattr(cp, dtype))
    arrs = [
        pa.array([1, 2, 3], type=arrow_type),
        pa.array([4, 5, 6], type=arrow_type),
    ]
    arrow_tbl = pa.Table.from_arrays(arrs, names=["a", "b"])
    tbl = plc.Table.from_arrow(arrow_tbl)

    rows, cols = tbl.num_rows(), tbl.num_columns()
    got = cp.empty((rows, cols), dtype=dtype, order="F")

    plc.reshape.table_to_array(
        tbl,
        got.data.ptr,
        got.nbytes,
    )

    expect = cp.array([[1, 4], [2, 5], [3, 6]], dtype=dtype)
    cp.testing.assert_array_equal(expect, got)
