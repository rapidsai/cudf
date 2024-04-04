# Copyright (c) 2024, NVIDIA CORPORATION.


import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc
from cudf._lib.types import dtype_to_pylibcudf_type

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


@pytest.fixture(scope="module")
def columns():
    return {
        "int8": plc.interop.from_arrow(pa.array([1, 2, 3, 4], type=pa.int8())),
        "int16": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.int16())
        ),
        "int32": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.int32())
        ),
        "int64": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.int64())
        ),
        "uint8": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.uint8())
        ),
        "uint16": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.uint16())
        ),
        "uint32": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.uint32())
        ),
        "uint64": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.uint64())
        ),
        "float32": plc.interop.from_arrow(
            pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float32())
        ),
        "float64": plc.interop.from_arrow(
            pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())
        ),
        "object": plc.interop.from_arrow(
            pa.array(["a", "b", "c", "d"], type=pa.string())
        ),
        "bool": plc.interop.from_arrow(
            pa.array([True, False, True, False], type=pa.bool_())
        ),
        "datetime64[ns]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("ns"))
        ),
        "datetime64[ms]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("ms"))
        ),
        "datetime64[us]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("us"))
        ),
        "datetime64[s]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.timestamp("s"))
        ),
        "timedelta64[ns]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("ns"))
        ),
        "timedelta64[ms]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("ms"))
        ),
        "timedelta64[us]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("us"))
        ),
        "timedelta64[s]": plc.interop.from_arrow(
            pa.array([1, 2, 3, 4], type=pa.duration("s"))
        ),
    }


@pytest.fixture(scope="module", params=LIBCUDF_SUPPORTED_TYPES)
def binop_lhs_ty(request):
    return request.param


@pytest.fixture(scope="module", params=LIBCUDF_SUPPORTED_TYPES)
def binop_rhs_ty(request):
    return request.param


@pytest.fixture(scope="module", params=LIBCUDF_SUPPORTED_TYPES)
def binop_out_ty(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=list(plc.binaryop.BinaryOperator.__members__.values()),
)
def binary_operators(request):
    return request.param


@pytest.fixture(scope="module")
def add_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    fail = False
    if not plc.binaryop._is_supported_binaryop(
        dtype_to_pylibcudf_type(binop_out_ty),
        dtype_to_pylibcudf_type(binop_lhs_ty),
        dtype_to_pylibcudf_type(binop_rhs_ty),
        plc.binaryop.BinaryOperator.ADD,
    ):
        fail = True
    return (binop_lhs_ty, binop_rhs_ty, binop_out_ty, fail)


def test_add(add_tests, columns):
    binop_lhs_ty, binop_rhs_ty, binop_out_ty, fail = add_tests
    lhs = columns[binop_lhs_ty]
    rhs = columns[binop_rhs_ty]
    pylibcudf_outty = dtype_to_pylibcudf_type(binop_out_ty)

    if not fail:
        expect_data = (
            plc.interop.to_arrow(lhs).to_numpy()
            + plc.interop.to_arrow(rhs).to_numpy()
        ).astype(binop_out_ty)
        expect = pa.array(expect_data)
        got = plc.binaryop.binary_operation(
            lhs, rhs, plc.binaryop.BinaryOperator.ADD, pylibcudf_outty
        )
        assert_column_eq(got, expect)
    else:
        with pytest.raises(TypeError):
            plc.binaryop.binary_operation(
                lhs, rhs, plc.binaryop.BinaryOperator.ADD, pylibcudf_outty
            )
