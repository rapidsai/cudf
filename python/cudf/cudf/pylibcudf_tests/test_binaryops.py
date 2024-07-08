# Copyright (c) 2024, NVIDIA CORPORATION.


from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


@pytest.fixture(params=[True, False])
def nulls(request):
    return request.param


@pytest.fixture
def pa_int_col(nulls):
    return pa.array([1, 2, 3 if not nulls else None, 4, 5], type=pa.int32())


@pytest.fixture
def plc_int_col(pa_int_col):
    return plc.interop.from_arrow(pa_int_col)


@pytest.fixture
def pa_uint_col(nulls):
    return pa.array([1, 2, 3 if not nulls else None, 4, 5], type=pa.uint32())


@pytest.fixture
def plc_uint_col(pa_uint_col):
    return plc.interop.from_arrow(pa_uint_col)


@pytest.fixture
def pa_float_col(nulls):
    return pa.array(
        [1.0, 2.0, 3.0 if not nulls else None, 4.0, 5.0], type=pa.float32()
    )


@pytest.fixture
def plc_float_col(pa_float_col):
    return plc.interop.from_arrow(pa_float_col)


@pytest.fixture
def pa_bool_col(nulls):
    return pa.array(
        [True, False, True if not nulls else None, False, True],
        type=pa.bool_(),
    )


@pytest.fixture
def plc_bool_col(pa_bool_col):
    return plc.interop.from_arrow(pa_bool_col)


@pytest.fixture
def pa_timestamp_col(nulls):
    return pa.array(
        [
            np.datetime64("2022-01-01"),
            np.datetime64("2022-01-02"),
            np.datetime64("2022-01-03") if not nulls else None,
            np.datetime64("2022-01-04"),
            np.datetime64("2022-01-05"),
        ],
        type=pa.timestamp("ns"),
    )


@pytest.fixture
def plc_timestamp_col(pa_timestamp_col):
    return plc.interop.from_arrow(pa_timestamp_col)


@pytest.fixture
def pa_duration_col(nulls):
    return pa.array(
        [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns") if not nulls else None,
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ],
        type=pa.duration("ns"),
    )


@pytest.fixture
def plc_duration_col(pa_duration_col):
    return plc.interop.from_arrow(pa_duration_col)


@pytest.fixture
def pa_decimal_col(nulls):
    return pa.array(
        [
            Decimal("1.23"),
            Decimal("4.56"),
            Decimal("7.89") if not nulls else None,
            Decimal("0.12"),
            Decimal("3.45"),
        ],
        type=pa.decimal128(9, 2),
    )


@pytest.fixture
def plc_decimal_col(pa_decimal_col):
    return plc.interop.from_arrow(pa_decimal_col)


def test_add(
    pa_int_col, pa_float_col, plc_int_col, plc_float_col, plc_duration_col
):
    expect = pa.compute.add(pa_int_col, pa_float_col).cast(pa.int32())
    got = plc.binaryop.binary_operation(
        plc_int_col,
        plc_float_col,
        plc.binaryop.BinaryOperator.ADD,
        plc.DataType(plc.TypeId.INT32),
    )

    assert_column_eq(expect, got)

    with pytest.raises(TypeError):
        plc.binaryop.binary_operation(
            plc_duration_col,
            plc_duration_col,
            plc.binaryop.BinaryOperator.ADD,
            plc.DataType(plc.TypeId.INT32),
        )


def test_sub(
    pa_int_col, pa_float_col, plc_int_col, plc_float_col, plc_duration_col
):
    expect = pa.compute.subtract(pa_int_col, pa_float_col).cast(pa.int32())
    got = plc.binaryop.binary_operation(
        plc_int_col,
        plc_float_col,
        plc.binaryop.BinaryOperator.SUB,
        plc.DataType(plc.TypeId.INT32),
    )

    assert_column_eq(expect, got)

    with pytest.raises(TypeError):
        plc.binaryop.binary_operation(
            plc_duration_col,
            plc_duration_col,
            plc.binaryop.BinaryOperator.SUB,
            plc.DataType(plc.TypeId.INT32),
        )


def test_mul(
    pa_int_col, pa_float_col, plc_int_col, plc_float_col, plc_duration_col
):
    expect = pa.compute.multiply(pa_int_col, pa_float_col).cast(pa.int32())
    got = plc.binaryop.binary_operation(
        plc_int_col,
        plc_float_col,
        plc.binaryop.BinaryOperator.MUL,
        plc.DataType(plc.TypeId.INT32),
    )

    assert_column_eq(expect, got)

    with pytest.raises(TypeError):
        plc.binaryop.binary_operation(
            plc_duration_col,
            plc_duration_col,
            plc.binaryop.BinaryOperator.MUL,
            plc.DataType(plc.TypeId.INT32),
        )


def test_div(
    pa_int_col, pa_float_col, plc_int_col, plc_float_col, plc_duration_col
):
    expect = pa.compute.divide(pa_int_col, pa_float_col).cast(pa.int32())
    got = plc.binaryop.binary_operation(
        plc_int_col,
        plc_float_col,
        plc.binaryop.BinaryOperator.DIV,
        plc.DataType(plc.TypeId.INT32),
    )

    assert_column_eq(expect, got)

    with pytest.raises(TypeError):
        plc.binaryop.binary_operation(
            plc_float_col,
            plc_duration_col,
            plc.binaryop.BinaryOperator.DIV,
            plc.DataType(plc.TypeId.INT32),
        )
