# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_array_eq, column_from_arrow

from cudf._lib import pylibcudf as plc
from cudf._lib.types import dtype_to_pylibcudf_type


@pytest.fixture(scope="module")
def columns():
    return {
        "int8": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.int8())),
        "int16": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.int16())),
        "int32": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.int32())),
        "int64": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.int64())),
        "uint8": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.uint8())),
        "uint16": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.uint16())),
        "uint32": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.uint32())),
        "uint64": column_from_arrow(pa.array([1, 2, 3, 4], type=pa.uint64())),
        "float32": column_from_arrow(
            pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32())
        ),
        "float64": column_from_arrow(
            pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())
        ),
    }


@pytest.mark.parametrize(
    "lty",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize(
    "rty",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
@pytest.mark.parametrize(
    "outty",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_add(columns, lty, rty, outty):
    lhs = columns[lty]
    rhs = columns[rty]

    pylibcudf_outty = dtype_to_pylibcudf_type(outty)

    expect = pa.array([2, 4, 6, 8], type=pa.type_for_alias(outty))
    got = plc.binaryop.binary_operation(
        lhs, rhs, plc.binaryop.BinaryOperator.ADD, pylibcudf_outty
    )

    assert_array_eq(got, expect)


def test_mismatched_sizes():
    lhs = column_from_arrow(pa.array([1, 2, 3, 4], type=pa.int32()))
    rhs = column_from_arrow(pa.array([1, 2, 3], type=pa.int32()))

    with pytest.raises(ValueError, match="Column sizes don't match"):
        plc.binaryop.binary_operation(
            lhs,
            rhs,
            plc.binaryop.BinaryOperator.ADD,
            dtype_to_pylibcudf_type("int32"),
        )
