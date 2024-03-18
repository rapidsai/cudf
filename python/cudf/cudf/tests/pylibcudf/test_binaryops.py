# Copyright (c) 2024, NVIDIA CORPORATION.

import itertools

import numpy as np
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
        "object": column_from_arrow(
            pa.array(["a", "b", "c", "d"], type=pa.string())
        ),
        "bool": column_from_arrow(
            pa.array([True, False, True, False], type=pa.bool_())
        ),
        "datetime64[ns]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("ns"))
        ),
        "datetime64[ms]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("ms"))
        ),
        "datetime64[us]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("us"))
        ),
        "datetime64[s]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("s"))
        ),
        "timedelta64[ns]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("ns"))
        ),
        "timedelta64[ms]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("ms"))
        ),
        "timedelta64[us]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("us"))
        ),
        "timedelta64[s]": column_from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("s"))
        ),
    }


LIBCUDF_SUPPORTED_TYPES = [
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
    "object",
    "bool",
    "datetime64[ns]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[s]",
    "timedelta64[ns]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[s]",
]


def generate_binaryops_tests():
    tests = []
    for op in plc.binaryop.BinaryOperator.__members__.values():
        for combination in itertools.combinations_with_replacement(
            LIBCUDF_SUPPORTED_TYPES, 3
        ):
            tests.append((*combination, op))
    return tests


@pytest.mark.parametrize("lty, rty, outty, op", generate_binaryops_tests())
def test_binaryops(columns, lty, rty, outty, op):
    lhs = columns[lty]
    rhs = columns[rty]
    pylibcudf_outty = dtype_to_pylibcudf_type(outty)

    if plc.binaryop._is_supported_operation(
        pylibcudf_outty,
        dtype_to_pylibcudf_type(lty),
        dtype_to_pylibcudf_type(rty),
        op,
    ):
        expect_data = np.array([2, 4, 6, 8]).astype(outty)

        expect = pa.array(expect_data, type=pa.from_numpy_dtype(outty))
        got = plc.binaryop.binary_operation(lhs, rhs, op, pylibcudf_outty)
        breakpoint()
        assert_array_eq(got, expect)
    else:
        with pytest.raises(TypeError):
            plc.binaryop.binary_operation(lhs, rhs, op, pylibcudf_outty)


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
