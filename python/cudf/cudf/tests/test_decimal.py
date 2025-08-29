# Copyright (c) 2021-2025, NVIDIA CORPORATION.

import decimal
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pytest

import cudf
from cudf.core.column import Decimal32Column, Decimal64Column, NumericalColumn
from cudf.core.dtypes import Decimal32Dtype, Decimal64Dtype
from cudf.testing import assert_eq
from cudf.testing._utils import (
    FLOAT_TYPES,
    INTEGER_TYPES,
    SIGNED_TYPES,
    expect_warning_if,
)


@pytest.mark.parametrize(
    "data_",
    [
        [Decimal("1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [Decimal("-1.1"), Decimal("2.2"), Decimal("3.3"), Decimal("4.4")],
        [1],
        [-1],
        [1, 2, 3, 4],
        [42, 17, 41],
        [1, 2, None, 4],
        [None, None, None],
        [],
    ],
)
@pytest.mark.parametrize(
    "typ_",
    [
        pa.decimal128(precision=4, scale=2),
        pa.decimal128(precision=5, scale=3),
        pa.decimal128(precision=6, scale=4),
    ],
)
@pytest.mark.parametrize("col", [Decimal32Column, Decimal64Column])
def test_round_trip_decimal_column(data_, typ_, col):
    pa_arr = pa.array(data_, type=typ_)
    col_32 = col.from_arrow(pa_arr)
    assert pa_arr.equals(col_32.to_arrow())


def test_from_arrow_max_precision_decimal64():
    with pytest.raises(ValueError):
        Decimal64Column.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal128(scale=0, precision=19))
        )


def test_from_arrow_max_precision_decimal32():
    with pytest.raises(ValueError):
        Decimal32Column.from_arrow(
            pa.array([1, 2, 3], type=pa.decimal128(scale=0, precision=10))
        )


@pytest.mark.parametrize("from_dtype", FLOAT_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(11, 4), Decimal64Dtype(18, 9)],
)
def test_typecast_from_float_to_decimal(request, from_dtype, to_dtype):
    data = cudf.Series(
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
    )
    request.applymarker(
        pytest.mark.xfail(
            from_dtype == np.dtype("float32") and to_dtype.precision > 12,
            reason="https://github.com/rapidsai/cudf/issues/14169",
        )
    )
    got = data.astype(from_dtype)

    pa_arr = got.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale)
    )
    expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("from_dtype", INTEGER_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    [Decimal64Dtype(9, 3), Decimal64Dtype(11, 4), Decimal64Dtype(18, 9)],
)
def test_typecast_from_int_to_decimal(from_dtype, to_dtype):
    data = cudf.Series(
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
    )
    got = data.astype(from_dtype)

    pa_arr = (
        got.to_arrow()
        .cast("float64")
        .cast(pa.decimal128(to_dtype.precision, to_dtype.scale))
    )
    expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "from_dtype",
    [
        Decimal64Dtype(7, 2),
        Decimal64Dtype(11, 4),
        Decimal64Dtype(18, 10),
        Decimal32Dtype(7, 2),
        Decimal32Dtype(5, 3),
        Decimal32Dtype(9, 5),
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        Decimal64Dtype(7, 2),
        Decimal64Dtype(18, 10),
        Decimal64Dtype(11, 4),
        Decimal32Dtype(7, 2),
        Decimal32Dtype(9, 5),
        Decimal32Dtype(5, 3),
    ],
)
def test_typecast_to_from_decimal(from_dtype, to_dtype):
    data = cudf.Series(
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
    )
    if from_dtype.scale > to_dtype.MAX_PRECISION:
        pytest.skip(
            "This is supposed to overflow because the representation value in "
            "the source exceeds the max representable in destination dtype."
        )
    s = data.astype(from_dtype)

    pa_arr = s.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale), safe=False
    )
    if isinstance(to_dtype, Decimal32Dtype):
        expected = cudf.Series._from_column(Decimal32Column.from_arrow(pa_arr))
    elif isinstance(to_dtype, Decimal64Dtype):
        expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    with expect_warning_if(to_dtype.scale < s.dtype.scale, UserWarning):
        got = s.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "from_dtype",
    [Decimal64Dtype(7, 2), Decimal64Dtype(11, 4), Decimal64Dtype(17, 10)],
)
@pytest.mark.parametrize("to_dtype", SIGNED_TYPES)
def test_typecast_from_decimal(from_dtype, to_dtype):
    data = cudf.Series(
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
    )
    got = data.astype(from_dtype)
    pa_arr = got.to_arrow().cast(to_dtype, safe=False)

    got = got.astype(to_dtype)
    expected = cudf.Series._from_column(NumericalColumn.from_arrow(pa_arr))

    assert_eq(got, expected)
    assert_eq(got.dtype, expected.dtype)


@pytest.mark.parametrize(
    "data, dtype, item, to, expect",
    [
        # scatter to a single index
        (
            ["1", "2", "3"],
            Decimal64Dtype(1, 0),
            Decimal(5),
            1,
            ["1", "5", "3"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            Decimal64Dtype(2, 1),
            Decimal("5.5"),
            1,
            ["1.5", "5.5", "3.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            Decimal64Dtype(5, 4),
            Decimal("5.0042"),
            1,
            ["1.0042", "5.0042", "3.0042"],
        ),
        # scatter via boolmask
        (
            ["1", "2", "3"],
            Decimal64Dtype(1, 0),
            Decimal(5),
            [True, False, True],
            ["5", "2", "5"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            Decimal64Dtype(2, 1),
            Decimal("5.5"),
            [True, True, True],
            ["5.5", "5.5", "5.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            Decimal64Dtype(5, 4),
            Decimal("5.0042"),
            [False, False, True],
            ["1.0042", "2.0042", "5.0042"],
        ),
        # We will allow assigning a decimal with less precision
        (
            ["1.00", "2.00", "3.00"],
            Decimal64Dtype(3, 2),
            Decimal(5),
            1,
            ["1.00", "5.00", "3.00"],
        ),
        # But not truncation
        (
            ["1", "2", "3"],
            Decimal64Dtype(1, 0),
            Decimal("5.5"),
            1,
            pa.lib.ArrowInvalid,
        ),
        # We will allow for setting scalars into decimal columns
        (["1", "2", "3"], Decimal64Dtype(1, 0), 5, 1, ["1", "5", "3"]),
        # But not if it has too many digits to fit the precision
        (["1", "2", "3"], Decimal64Dtype(1, 0), 50, 1, pa.lib.ArrowInvalid),
    ],
)
def test_series_setitem_decimal(data, dtype, item, to, expect):
    data = cudf.Series([Decimal(x) for x in data], dtype=dtype)

    if expect is pa.lib.ArrowInvalid:
        with pytest.raises(expect):
            data[to] = item
        return
    else:
        expect = cudf.Series([Decimal(x) for x in expect], dtype=dtype)
        data[to] = item
        assert_eq(data, expect)


@pytest.mark.parametrize(
    "input_obj", [[decimal.Decimal(1), cudf.NA, decimal.Decimal(3)]]
)
def test_series_construction_with_nulls(input_obj):
    expect = pa.array(input_obj, from_pandas=True)
    got = cudf.Series(input_obj).to_arrow()

    assert expect == got


@pytest.mark.parametrize(
    "data",
    [
        [(["1", "2", "3"], cudf.Decimal64Dtype(1, 0))],
        [
            (["1", "2", "3"], cudf.Decimal64Dtype(1, 0)),
            (["1.0", "2.0", "3.0"], cudf.Decimal64Dtype(2, 1)),
            (["10.1", "20.2", "30.3"], cudf.Decimal64Dtype(3, 1)),
        ],
        [
            (["1", None, "3"], cudf.Decimal64Dtype(1, 0)),
            (["1.0", "2.0", None], cudf.Decimal64Dtype(2, 1)),
            ([None, "20.2", "30.3"], cudf.Decimal64Dtype(3, 1)),
        ],
    ],
)
def test_serialize_decimal_columns(data):
    df = cudf.DataFrame(
        {
            str(i): cudf.Series(
                [Decimal(x) if x is not None else x for x in values],
                dtype=dtype,
            )
            for i, (values, dtype) in enumerate(data)
        }
    )
    recreated = df.__class__.deserialize(*df.serialize())
    assert_eq(recreated, df)


def test_decimal_invalid_precision():
    with pytest.raises(pa.ArrowInvalid):
        _ = cudf.Series([10, 20, 30], dtype=cudf.Decimal64Dtype(2, 2))

    with pytest.raises(pa.ArrowInvalid):
        _ = cudf.Series([Decimal("300")], dtype=cudf.Decimal64Dtype(2, 1))


def test_decimal_overflow():
    s = cudf.Series([Decimal("0.0009384233522166997927180531650178250")])
    result = s * s
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=37), result.dtype)

    s = cudf.Series([1, 2], dtype=cudf.Decimal128Dtype(precision=38, scale=0))
    result = s * Decimal("1.0")
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=1), result.dtype)


def test_decimal_binop_upcast_operands():
    ser1 = cudf.Series([0.51, 1.51, 2.51]).astype(cudf.Decimal64Dtype(18, 2))
    ser2 = cudf.Series([0.90, 0.96, 0.99]).astype(cudf.Decimal128Dtype(19, 2))
    result = ser1 + ser2
    expected = cudf.Series([1.41, 2.47, 3.50]).astype(
        cudf.Decimal128Dtype(20, 2)
    )
    assert_eq(result, expected)
