# Copyright (c) 2021, NVIDIA CORPORATION.

from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

import cudf
from cudf.core.column import DecimalColumn, NumericalColumn
from cudf.core.dtypes import Decimal64Dtype
from cudf.tests.utils import (
    FLOAT_TYPES,
    INTEGER_TYPES,
    NUMERIC_TYPES,
    assert_eq,
)


@pytest.mark.parametrize(
    "data",
    [
        [Decimal("1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [Decimal("-1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [1],
        [-1],
        [1, 2, 3, 4],
        [42, 1729, 4104],
        [1, 2, None, 4],
        [None, None, None],
        [],
    ],
)
@pytest.mark.parametrize(
    "typ",
    [
        pa.decimal128(precision=4, scale=2),
        pa.decimal128(precision=5, scale=3),
        pa.decimal128(precision=6, scale=4),
    ],
)
def test_round_trip_decimal_column(data, typ):
    pa_arr = pa.array(data, type=typ)
    col = DecimalColumn.from_arrow(pa_arr)
    assert pa_arr.equals(col.to_arrow())


def test_from_arrow_max_precision():
    with pytest.raises(ValueError):
        DecimalColumn.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal128(scale=0, precision=19))
        )


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(
            [
                14.12302,
                97938.2,
                np.nan,
                0.0,
                -8.302014,
                np.nan,
                94.31304,
                -112.2314,
                0.3333333,
                np.nan,
            ]
        ),
    ],
)
@pytest.mark.parametrize("from_dtype", FLOAT_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(11, 4), Decimal64Dtype(18, 9)],
)
def test_typecast_from_float_to_decimal(data, from_dtype, to_dtype):
    got = data.astype(from_dtype)

    pa_arr = got.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale)
    )
    expected = cudf.Series(DecimalColumn.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(
            [
                14.12302,
                38.2,
                np.nan,
                0.0,
                -8.302014,
                np.nan,
                94.31304,
                np.nan,
                -112.2314,
                0.3333333,
                np.nan,
            ]
        ),
    ],
)
@pytest.mark.parametrize("from_dtype", INTEGER_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    [Decimal64Dtype(9, 3), Decimal64Dtype(11, 4), Decimal64Dtype(18, 9)],
)
def test_typecast_from_int_to_decimal(data, from_dtype, to_dtype):
    got = data.astype(from_dtype)

    pa_arr = (
        got.to_arrow()
        .cast("float64")
        .cast(pa.decimal128(to_dtype.precision, to_dtype.scale))
    )
    expected = cudf.Series(DecimalColumn.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(
            [
                14.12309,
                2.343942,
                np.nan,
                0.0,
                -8.302082,
                np.nan,
                94.31308,
                -112.2364,
                -8.029972,
                np.nan,
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "from_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(11, 4), Decimal64Dtype(18, 10)],
)
@pytest.mark.parametrize(
    "to_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(18, 10), Decimal64Dtype(11, 4)],
)
def test_typecast_to_from_decimal(data, from_dtype, to_dtype):
    got = data.astype(from_dtype)

    pa_arr = got.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale), safe=False
    )
    expected = cudf.Series(DecimalColumn.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        cudf.Series(
            [
                14.12309,
                2.343942,
                np.nan,
                0.0,
                -8.302082,
                np.nan,
                94.31308,
                -112.2364,
                -8.029972,
                np.nan,
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "from_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(11, 4), Decimal64Dtype(17, 10)],
)
@pytest.mark.parametrize("to_dtype", NUMERIC_TYPES)
def test_typecast_from_decimal(data, from_dtype, to_dtype):
    got = data.astype(from_dtype)
    pa_arr = got.to_arrow().cast(to_dtype, safe=False)

    got = got.astype(to_dtype)
    expected = cudf.Series(NumericalColumn.from_arrow(pa_arr))

    assert_eq(got, expected)
