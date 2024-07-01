# Copyright (c) 2024, NVIDIA CORPORATION.


import warnings

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc
from cudf._lib.types import dtype_to_pylibcudf_type

LIBCUDF_SUPPORTED_TYPES = [
    #    "int8",
    #    "int16",
    #    "int32",
    "int64",
    #    "uint8",
    #    "uint16",
    #    "uint32",
    "uint64",
    #    "float32",
    "float64",
    "object",
    "bool",
    "datetime64[ns]",
    #    "datetime64[ms]",
    #    "datetime64[us]",
    #    "datetime64[s]",
    "timedelta64[ns]",
    #    "timedelta64[ms]",
    #    "timedelta64[us]",
    #    "timedelta64[s]",
]

BINARY_OPS = list(plc.binaryop.BinaryOperator.__members__.values())


@pytest.fixture(scope="module")
def columns():
    return {
        "int8": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.int8())
        ),
        "int16": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.int16())
        ),
        "int32": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.int32())
        ),
        "int64": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.int64())
        ),
        "uint8": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.uint8())
        ),
        "uint16": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.uint16())
        ),
        "uint32": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.uint32())
        ),
        "uint64": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.uint64())
        ),
        "float32": plc.interop.from_arrow(
            pa.array([1.0, None, 3.0, None], type=pa.float32())
        ),
        "float64": plc.interop.from_arrow(
            pa.array([1.0, 2.0, None, None], type=pa.float64())
        ),
        "object": plc.interop.from_arrow(
            pa.array(["a", None, "c", None], type=pa.string())
        ),
        "bool": plc.interop.from_arrow(
            pa.array([True, False, None, None], type=pa.bool_())
        ),
        "datetime64[ns]": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.timestamp("ns"))
        ),
        "datetime64[ms]": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.timestamp("ms"))
        ),
        "datetime64[us]": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.timestamp("us"))
        ),
        "datetime64[s]": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.timestamp("s"))
        ),
        "timedelta64[ns]": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.duration("ns"))
        ),
        "timedelta64[ms]": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.duration("ms"))
        ),
        "timedelta64[us]": plc.interop.from_arrow(
            pa.array([1, None, 3, None], type=pa.duration("us"))
        ),
        "timedelta64[s]": plc.interop.from_arrow(
            pa.array([1, 2, None, None], type=pa.duration("s"))
        ),
    }


@pytest.fixture(scope="module")
def null_columns(columns):
    breakpoint()


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
    params=BINARY_OPS,
)
def binary_operator(request):
    return request.param


def make_test(binop_lhs_ty, binop_rhs_ty, binop_out_ty, binary_operator):
    fail = False
    if not plc.binaryop._is_supported_binaryop(
        dtype_to_pylibcudf_type(binop_out_ty),
        dtype_to_pylibcudf_type(binop_lhs_ty),
        dtype_to_pylibcudf_type(binop_rhs_ty),
        binary_operator,
    ):
        fail = True
    return (binop_lhs_ty, binop_rhs_ty, binop_out_ty, fail)


def _test_binaryop_inner(test, columns, pyop, cuop):
    binop_lhs_ty, binop_rhs_ty, binop_out_ty, fail = test
    lhs = columns[binop_lhs_ty]
    rhs = columns[binop_rhs_ty]
    pylibcudf_outty = dtype_to_pylibcudf_type(binop_out_ty)
    if not fail:

        def op(x, y, dtype):
            result = pyop(x, y)
            if result is None:
                return result
            else:
                if hasattr(result, "to_numpy"):
                    return result.to_numpy().astype(dtype)
                else:
                    return np.dtype(dtype).type(result).astype(dtype)

        expect_data = [
            op(x, y, binop_out_ty)
            for x, y in zip(
                plc.interop.to_arrow(lhs).to_pylist(),
                plc.interop.to_arrow(rhs).to_pylist(),
            )
        ]
        expect = pa.array(expect_data, type=pa.from_numpy_dtype(binop_out_ty))
        got = plc.binaryop.binary_operation(lhs, rhs, cuop, pylibcudf_outty)
        assert_column_eq(got, expect)
    else:
        with pytest.raises(TypeError):
            plc.binaryop.binary_operation(lhs, rhs, cuop, pylibcudf_outty)


@pytest.fixture(scope="module")
def add_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.ADD,
    )


@pytest.fixture(scope="module")
def sub_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.SUB,
    )


@pytest.fixture(scope="module")
def mul_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.MUL,
    )


@pytest.fixture(scope="module")
def div_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.DIV,
    )


@pytest.fixture(scope="module")
def true_div_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.TRUE_DIV,
    )


@pytest.fixture(scope="module")
def floor_div_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.FLOOR_DIV,
    )


@pytest.fixture(scope="module")
def mod_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.MOD,
    )


@pytest.fixture(scope="module")
def pmod_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.PMOD,
    )


@pytest.fixture(scope="module")
def pymod_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    # TODO
    pass


@pytest.fixture(scope="module")
def pow_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.POW,
    )


@pytest.fixture(scope="module")
def int_pow_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.INT_POW,
    )


@pytest.fixture(scope="module")
def log_base_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.LOG_BASE,
    )


@pytest.fixture(scope="module")
def atan2_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.ATAN2,
    )


@pytest.fixture(scope="module")
def shift_left_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.SHIFT_LEFT,
    )


@pytest.fixture(scope="module")
def shift_right_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.SHIFT_RIGHT,
    )


@pytest.fixture(scope="module")
def shift_right_unsigned_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
    )


@pytest.fixture(scope="module")
def bitwise_and_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.BITWISE_AND,
    )


@pytest.fixture(scope="module")
def bitwise_or_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.BITWISE_OR,
    )


@pytest.fixture(scope="module")
def logical_and_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.LOGICAL_AND,
    )


@pytest.fixture(scope="module")
def logical_or_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.LOGICAL_OR,
    )


@pytest.fixture(scope="module")
def equal_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.EQUAL,
    )


@pytest.fixture(scope="module")
def not_equal_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NOT_EQUAL,
    )


@pytest.fixture(scope="module")
def less_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.LESS,
    )


@pytest.fixture(scope="module")
def greater_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.GREATER,
    )


@pytest.fixture(scope="module")
def less_equal_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.LESS_EQUAL,
    )


@pytest.fixture(scope="module")
def greater_equal_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
    )


@pytest.fixture(scope="module")
def null_equals_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NULL_EQUALS,
    )


@pytest.fixture(scope="module")
def null_max_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NULL_MAX,
    )


@pytest.fixture(scope="module")
def null_min_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NULL_MIN,
    )


@pytest.fixture(scope="module")
def generic_binary_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.GENERIC_BINARY,
    )


@pytest.fixture(scope="module")
def null_logical_and_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
    )


@pytest.fixture(scope="module")
def null_logical_or_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    )


@pytest.fixture(scope="module")
def invalid_binary_tests(binop_lhs_ty, binop_rhs_ty, binop_out_ty):
    return make_test(
        binop_lhs_ty,
        binop_rhs_ty,
        binop_out_ty,
        plc.binaryop.BinaryOperator.INVALID_BINARY,
    )


def test_add(add_tests, columns):
    def add(x, y):
        if x is None or y is None:
            return None
        return x + y

    _test_binaryop_inner(
        add_tests, columns, add, plc.binaryop.BinaryOperator.ADD
    )


def test_sub(sub_tests, columns):
    def sub(x, y):
        if x is None or y is None:
            return None
        return x - y

    _test_binaryop_inner(
        sub_tests, columns, sub, plc.binaryop.BinaryOperator.SUB
    )


def test_mul(mul_tests, columns):
    def mul(x, y):
        if x is None or y is None:
            return None
        return x * y

    _test_binaryop_inner(
        mul_tests, columns, mul, plc.binaryop.BinaryOperator.MUL
    )


def test_div(div_tests, columns):
    def div(x, y):
        if x is None or y is None:
            return None
        return x / y

    _test_binaryop_inner(
        div_tests, columns, div, plc.binaryop.BinaryOperator.DIV
    )


def test_true_div(true_div_tests, columns):
    _test_binaryop_inner(
        true_div_tests,
        columns,
        np.true_divide,
        plc.binaryop.BinaryOperator.TRUE_DIV,
    )


def test_floor_div(floor_div_tests, columns):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", message="divide by zero")
        _test_binaryop_inner(
            floor_div_tests,
            columns,
            np.floor_divide,
            plc.binaryop.BinaryOperator.FLOOR_DIV,
        )


def test_mod(mod_tests, columns):
    _test_binaryop_inner(
        mod_tests, columns, np.mod, plc.binaryop.BinaryOperator.MOD
    )


def test_pmod(pmod_tests, columns):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", message="divide by zero")
        warnings.filterwarnings("ignore", message="invalid value")
        _test_binaryop_inner(
            pmod_tests,
            columns,
            lambda x, y: (x % y + y) % x,
            plc.binaryop.BinaryOperator.PMOD,
        )


def test_pow(pow_tests, columns):
    _test_binaryop_inner(
        pow_tests, columns, np.power, plc.binaryop.BinaryOperator.POW
    )


def test_int_pow(int_pow_tests, columns):
    _test_binaryop_inner(
        int_pow_tests, columns, np.power, plc.binaryop.BinaryOperator.INT_POW
    )


def test_log_base(log_base_tests, columns):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", message="divide by zero")
        warnings.filterwarnings("ignore", message="invalid value")
        _test_binaryop_inner(
            log_base_tests,
            columns,
            lambda x, y: np.log(y) / np.log(x),
            plc.binaryop.BinaryOperator.LOG_BASE,
        )


def test_atan2(atan2_tests, columns):
    _test_binaryop_inner(
        atan2_tests, columns, np.arctan2, plc.binaryop.BinaryOperator.ATAN2
    )


def test_shift_left(shift_left_tests, columns):
    _test_binaryop_inner(
        shift_left_tests,
        columns,
        np.left_shift,
        plc.binaryop.BinaryOperator.SHIFT_LEFT,
    )


def test_shift_right(shift_right_tests, columns):
    _test_binaryop_inner(
        shift_right_tests,
        columns,
        np.right_shift,
        plc.binaryop.BinaryOperator.SHIFT_RIGHT,
    )


def test_shift_right_unsigned(shift_right_unsigned_tests, columns):
    _test_binaryop_inner(
        shift_right_unsigned_tests,
        columns,
        lambda x, y: np.right_shift(
            x.astype(np.int64), y.astype(np.int64)
        ).astype(x.dtype),
        plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
    )


def test_bitwise_and(bitwise_and_tests, columns):
    _test_binaryop_inner(
        bitwise_and_tests,
        columns,
        np.bitwise_and,
        plc.binaryop.BinaryOperator.BITWISE_AND,
    )


def test_bitwise_or(bitwise_or_tests, columns):
    _test_binaryop_inner(
        bitwise_or_tests,
        columns,
        np.bitwise_or,
        plc.binaryop.BinaryOperator.BITWISE_OR,
    )


def test_bitwise_xor(bitwise_xor_tests, columns):
    _test_binaryop_inner(
        bitwise_xor_tests,
        columns,
        np.bitwise_xor,
        plc.binaryop.BinaryOperator.BITWISE_XOR,
    )


def test_logical_and(logical_and_tests, columns):
    _test_binaryop_inner(
        logical_and_tests,
        columns,
        np.logical_and,
        plc.binaryop.BinaryOperator.LOGICAL_AND,
    )


def test_logical_or(logical_or_tests, columns):
    _test_binaryop_inner(
        logical_or_tests,
        columns,
        np.logical_or,
        plc.binaryop.BinaryOperator.LOGICAL_OR,
    )


def test_equal(equal_tests, columns):
    _test_binaryop_inner(
        equal_tests, columns, np.equal, plc.binaryop.BinaryOperator.EQUAL
    )


def test_not_equal(not_equal_tests, columns):
    _test_binaryop_inner(
        not_equal_tests,
        columns,
        np.not_equal,
        plc.binaryop.BinaryOperator.NOT_EQUAL,
    )


def test_less(less_tests, columns):
    _test_binaryop_inner(
        less_tests, columns, np.less, plc.binaryop.BinaryOperator.LESS
    )


def test_greater(greater_tests, columns):
    _test_binaryop_inner(
        greater_tests, columns, np.greater, plc.binaryop.BinaryOperator.GREATER
    )


def test_less_equal(less_equal_tests, columns):
    _test_binaryop_inner(
        less_equal_tests,
        columns,
        np.less_equal,
        plc.binaryop.BinaryOperator.LESS_EQUAL,
    )


def test_greater_equal(greater_equal_tests, columns):
    _test_binaryop_inner(
        greater_equal_tests,
        columns,
        np.greater_equal,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
    )


def test_null_equals(null_equals_tests, columns):
    def numpy_null_equals(x, y):
        # TODO
        pass

    _test_binaryop_inner(
        null_equals_tests,
        columns,
        numpy_null_equals,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
    )


def test_null_max(null_max_tests, columns):
    # TODO
    pass


def test_null_min(null_min_tests, columns):
    # TODO
    pass


def test_generic_binary(generic_binary_tests, columns):
    # TODO
    pass


def test_null_logical_and(null_logical_and_tests, columns):
    # TODO
    pass


def test_null_logical_or(null_logical_or_tests, columns):
    # TODO
    pass


def test_invalid_binary(invalid_binary_tests, columns):
    # TODO
    pass
