# Copyright (c) 2024-2025, NVIDIA CORPORATION.

import math

import numba
import pyarrow as pa
import pytest
from numba import cuda
from utils import assert_column_eq

import pylibcudf as plc


def test_nans_to_nulls(has_nans):
    if has_nans:
        values = [1, float("nan"), float("nan"), None, 3, None]
    else:
        values = [1, 4, 5, None, 3, None]

    replaced = [
        None if (v is None or (v is not None and math.isnan(v))) else v
        for v in values
    ]

    h_input = pa.array(values, type=pa.float32())
    input = plc.interop.from_arrow(h_input)
    assert input.null_count() == h_input.null_count
    expect = pa.array(replaced, type=pa.float32())

    mask, null_count = plc.transform.nans_to_nulls(input)

    assert null_count == expect.null_count
    got = input.with_mask(mask, null_count)

    assert_column_eq(expect, got)


def test_bools_to_mask_roundtrip():
    pa_array = pa.array([True, None, False])
    plc_input = plc.interop.from_arrow(pa_array)
    mask, result_null_count = plc.transform.bools_to_mask(plc_input)

    assert result_null_count == 2
    result = plc_input.with_mask(mask, result_null_count)
    assert_column_eq(pa.array([True, None, None]), result)

    plc_output = plc.transform.mask_to_bools(mask.ptr, 0, len(pa_array))
    result_pa = plc.interop.to_arrow(plc_output)
    expected_pa = pa.chunked_array([[True, False, False]])
    assert result_pa.equals(expected_pa)


def test_encode():
    pa_table = pa.table({"a": [1, 3, 4], "b": [1, 2, 4]})
    plc_input = plc.interop.from_arrow(pa_table)
    result_table, result_column = plc.transform.encode(plc_input)
    pa_table_result = plc.interop.to_arrow(result_table)
    pa_column_result = plc.interop.to_arrow(result_column)

    pa_table_expected = pa.table(
        [[1, 3, 4], [1, 2, 4]],
        schema=pa.schema(
            [
                pa.field("", pa.int64(), nullable=False),
                pa.field("", pa.int64(), nullable=False),
            ]
        ),
    )
    assert pa_table_result.equals(pa_table_expected)

    pa_column_expected = pa.chunked_array([[0, 1, 2]], type=pa.int32())
    assert pa_column_result.equals(pa_column_expected)


def test_one_hot_encode():
    pa_column = pa.array([1, 2, 3])
    pa_categories = pa.array([0, 0, 0])
    plc_input = plc.interop.from_arrow(pa_column)
    plc_categories = plc.interop.from_arrow(pa_categories)
    plc_table = plc.transform.one_hot_encode(plc_input, plc_categories)
    result = plc.interop.to_arrow(plc_table)
    expected = pa.table(
        [[False] * 3] * 3,
        schema=pa.schema([pa.field("", pa.bool_(), nullable=False)] * 3),
    )
    assert result.equals(expected)


def test_transform_udf():
    @cuda.jit(device=True)
    def op(a, b, c):
        return (a + b) * c

    if not plc.jit.is_runtime_jit_supported():
        pytest.skip("Skipping tests that require runtime JIT support")

    ptx, _ = cuda.compile_ptx_for_current_device(
        op, (numba.float64, numba.float64, numba.float64), device=True
    )

    A = 5.0
    B = 20.0
    C = 0.5

    a = pa.array([A] * 100)
    b = pa.array([B] * 100)
    c = pa.array([C])
    expected = pa.array([(A + B) * C] * 100)
    result = plc.transform.transform(
        [
            plc.interop.from_arrow(a),
            plc.interop.from_arrow(b),
            plc.interop.from_arrow(c),
        ],
        transform_udf=ptx,
        output_type=plc.DataType(plc.TypeId.FLOAT64),
        is_ptx=True,
    )
    assert_column_eq(expected, result)
