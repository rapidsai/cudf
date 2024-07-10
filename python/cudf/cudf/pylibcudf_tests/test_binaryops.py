# Copyright (c) 2024, NVIDIA CORPORATION.


import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def idfn(param):
    ltype, rtype, outtype = param
    return f"{ltype}-{rtype}-{outtype}"


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

    expect = [
        pyop(x, y) for x, y in zip(lhs_py.to_pylist(), rhs_py.to_pylist())
    ]
    expect = pa.array(expect, type=outty_py)
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
    def add(x, y):
        if x is None or y is None:
            return None
        return x + y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        add,
        plc.binaryop.BinaryOperator.ADD,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_sub(pa_data, plc_data):
    def sub(x, y):
        if x is None or y is None:
            return None
        return x - y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        sub,
        plc.binaryop.BinaryOperator.SUB,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_mul(pa_data, plc_data):
    def mul(x, y):
        if x is None or y is None:
            return None
        return x * y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        mul,
        plc.binaryop.BinaryOperator.MUL,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_div(pa_data, plc_data):
    def div(x, y):
        if x is None or y is None:
            return None
        return x / y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        div,
        plc.binaryop.BinaryOperator.DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_floordiv(pa_data, plc_data):
    def floordiv(x, y):
        if x is None or y is None:
            return None
        return x // y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        floordiv,
        plc.binaryop.BinaryOperator.FLOOR_DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_truediv(pa_data, plc_data):
    def truediv(x, y):
        if x is None or y is None:
            return None
        return x / y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        truediv,
        plc.binaryop.BinaryOperator.TRUE_DIV,
    )


@pytest.mark.parametrize(
    "pa_data",
    [("int64", "int64", "int64"), ("int64", "float64", "float64")],
    indirect=True,
    ids=idfn,
)
def test_mod(pa_data, plc_data):
    def mod(x, y):
        if x is None or y is None:
            return None
        return x % y

    _test_binaryop_inner(
        pa_data,
        plc_data,
        mod,
        plc.binaryop.BinaryOperator.MOD,
    )
