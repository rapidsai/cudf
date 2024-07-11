# Copyright (c) 2024, NVIDIA CORPORATION.


import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def idfn(param):
    ltype, rtype, outtype = param
    return "-".join(map(str, param))


@pytest.fixture(params=[True, False], ids=["nulls", "no_nulls"])
def nulls(request):
    return request.param


@pytest.fixture
def pa_data(request, nulls):
    ltype, rtype, outtype = request.param
    values = make_col(ltype, nulls), make_col(rtype, nulls), outtype
    return values


@pytest.fixture
def plc_data(pa_data):
    lhs, rhs, outtype = pa_data
    return (
        plc.interop.from_arrow(lhs),
        plc.interop.from_arrow(rhs),
        plc.interop.from_arrow(pa.from_numpy_dtype(np.dtype(outtype))),
    )


def make_col(dtype, nulls):
    if dtype == "int64":
        data = [1, 2, 3, 4, 5]
        pa_type = pa.int32()
    elif dtype == "uint64":
        data = [1, 2, 3, 4, 5]
        pa_type = pa.uint32()
    elif dtype == "float64":
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        pa_type = pa.float32()
    elif dtype == "bool":
        data = [True, False, True, False, True]
        pa_type = pa.bool_()
    elif dtype == "timestamp64[ns]":
        data = [
            np.datetime64("2022-01-01"),
            np.datetime64("2022-01-02"),
            np.datetime64("2022-01-03"),
            np.datetime64("2022-01-04"),
            np.datetime64("2022-01-05"),
        ]
        pa_type = pa.timestamp("ns")
    elif dtype == "timedelta64[ns]":
        data = [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns"),
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ]
        pa_type = pa.duration("ns")
    else:
        raise ValueError("Unsupported dtype")

    if nulls:
        data[3] = None

    return pa.array(data, type=pa_type)


def _test_binaryop_inner(pa_data, plc_data, pyop, plc_op):
    lhs_py, rhs_py, outty_py = pa_data
    lhs_plc, rhs_plc, outty_plc = plc_data

    def get_result():
        return plc.binaryop.binary_operation(
            lhs_plc,
            rhs_plc,
            plc_op,
            outty_plc,
        )

    if not plc.binaryop.is_supported_operation(
        outty_plc, lhs_plc.type(), rhs_plc.type(), plc_op
    ):
        with pytest.raises(TypeError):
            get_result()
        return
    expect = pyop(lhs_py, rhs_py).cast(outty_py)
    got = get_result()
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_add(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.add,
        plc.binaryop.BinaryOperator.ADD,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_sub(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.subtract,
        plc.binaryop.BinaryOperator.SUB,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_mul(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.multiply,
        plc.binaryop.BinaryOperator.MUL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_div(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.divide,
        plc.binaryop.BinaryOperator.DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "timedelta64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_truediv(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.divide,
        plc.binaryop.BinaryOperator.TRUE_DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_floordiv(pa_data, plc_data):
    def floordiv(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def slr_func(x, y):
            if x is None or y is None:
                return None
            return x // y

        return pa.array([slr_func(x, y) for x, y in zip(x, y)])

    _test_binaryop_inner(
        pa_data,
        plc_data,
        floordiv,
        plc.binaryop.BinaryOperator.FLOOR_DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_mod(pa_data, plc_data):
    def mod(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def slr_func(x, y):
            if x is None or y is None:
                return None
            return x % y

        return pa.array([slr_func(x, y) for x, y in zip(x, y)])

    _test_binaryop_inner(
        pa_data,
        plc_data,
        mod,
        plc.binaryop.BinaryOperator.MOD,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_pmod(pa_data, plc_data):
    def pmod(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def slr_func(x, y):
            if x is None or y is None:
                return None
            return (x % y + y) % y

        return pa.array([slr_func(x, y) for x, y in zip(x, y)])

    _test_binaryop_inner(
        pa_data,
        plc_data,
        pmod,
        plc.binaryop.BinaryOperator.PMOD,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_pymod(pa_data, plc_data):
    def pymod(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def slr_func(x, y):
            if x is None or y is None:
                return None
            return x % y

        return pa.array([slr_func(x, y) for x, y in zip(x, y)])

    _test_binaryop_inner(
        pa_data,
        plc_data,
        pymod,
        plc.binaryop.BinaryOperator.PYMOD,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "timedelta64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_pow(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.power,
        plc.binaryop.BinaryOperator.POW,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_int_pow(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.power,
        plc.binaryop.BinaryOperator.INT_POW,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("float64", "float64", "float64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "timedelta64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_log_base(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.logb,
        plc.binaryop.BinaryOperator.LOG_BASE,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("float64", "float64", "float64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "timedelta64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_atan2(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.atan2,
        plc.binaryop.BinaryOperator.ATAN2,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_shift_left(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.shift_left,
        plc.binaryop.BinaryOperator.SHIFT_LEFT,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_shift_right(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.shift_right,
        plc.binaryop.BinaryOperator.SHIFT_RIGHT,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_shift_right_unsigned(pa_data, plc_data):
    def shift_right_unsigned(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def logical_right_shift(x, y):
            if x is None or y is None:
                return None
            unsigned_x = np.uint32(x)
            result = unsigned_x >> y
            return result

        return pa.array([logical_right_shift(x, y) for x, y in zip(x, y)])

    _test_binaryop_inner(
        pa_data,
        plc_data,
        shift_right_unsigned,
        plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_bitwise_and(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.bit_wise_and,
        plc.binaryop.BinaryOperator.BITWISE_AND,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_bitwise_or(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.bit_wise_or,
        plc.binaryop.BinaryOperator.BITWISE_OR,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "datetime64[ns]"),
    ],
    indirect=True,
    ids=idfn,
)
def test_bitwise_xor(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.bit_wise_xor,
        plc.binaryop.BinaryOperator.BITWISE_XOR,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_logical_and(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.and_,
        plc.binaryop.BinaryOperator.LOGICAL_AND,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_logical_or(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.or_,
        plc.binaryop.BinaryOperator.LOGICAL_OR,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_equal(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.equal,
        plc.binaryop.BinaryOperator.EQUAL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_not_equal(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.not_equal,
        plc.binaryop.BinaryOperator.NOT_EQUAL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_less(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.less,
        plc.binaryop.BinaryOperator.LESS,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_greater(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.greater,
        plc.binaryop.BinaryOperator.GREATER,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_less_equal(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.less_equal,
        plc.binaryop.BinaryOperator.LESS_EQUAL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_greater_equal(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.greater_equal,
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_equals(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.equal,
        plc.binaryop.BinaryOperator.NULL_EQUALS,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "datetime64[ns]"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_max(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.max_element_wise,
        plc.binaryop.BinaryOperator.NULL_MAX,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "datetime64[ns]"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_min(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.min_element_wise,
        plc.binaryop.BinaryOperator.NULL_MIN,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
        ("int64", "int64", "int64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_not_equals(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.not_equal,
        plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_generic_binary(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        None,
        plc.binaryop.BinaryOperator.GENERIC_BINARY,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_logical_and(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.and_,
        plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_null_logical_or(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        pa.compute.or_,
        plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
    )


@pytest.mark.parametrize(
    "pa_data",
    [
        ("int64", "int64", "int64"),
        ("int64", "float64", "float64"),
    ],
    indirect=True,
    ids=idfn,
)
def test_invalid_binary(pa_data, plc_data):
    _test_binaryop_inner(
        pa_data,
        plc_data,
        None,
        plc.binaryop.BinaryOperator.INVALID_BINARY,
    )
