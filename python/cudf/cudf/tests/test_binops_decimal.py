# Copyright (c) 2021, NVIDIA CORPORATION.

import pytest
import operator
import cudf 
import numpy as np
import decimal
from cudf.tests import utils

def _decimal_series(input, dtype):
    return cudf.Series(
        [x if x is None else decimal.Decimal(x) for x in input], dtype=dtype,
    )

@pytest.mark.parametrize(
    "args",
    [
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
        ),
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", "3.005"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["100.1", "200.2"],
            cudf.Decimal64Dtype(scale=3, precision=9),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["99.9", "199.8"],
            cudf.Decimal64Dtype(scale=3, precision=9),
        ),
        (
            operator.mul,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "6.0"],
            cudf.Decimal64Dtype(scale=5, precision=7),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["10.0", "40.0"],
            cudf.Decimal64Dtype(scale=1, precision=8),
        ),
        (
            operator.mul,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-3, precision=4),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=3),
            ["343.0", "1000.0"],
            cudf.Decimal64Dtype(scale=0, precision=8),
        ),
        (
            operator.add,
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["3.0", None, "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
        ),
        (
            operator.add,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=2),
            ["1.5", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=5, precision=7),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["10.0", None],
            cudf.Decimal64Dtype(scale=1, precision=8),
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.18", "0.21"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1800", "0.2100"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["100", None],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["200", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, False],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, True],
            bool,
        ),
    ],
)
def test_binops_decimal(args):
    op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype = args

    a = _decimal_series(lhs, l_dtype)
    b = _decimal_series(rhs, r_dtype)
    expect = (
        _decimal_series(expect, expect_dtype)
        if isinstance(expect_dtype, cudf.Decimal64Dtype)
        else cudf.Series(expect, dtype=expect_dtype)
    )

    got = op(a, b)
    assert expect.dtype == got.dtype
    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "args",
    [
        (
            operator.eq,
            ["100", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.000", "42.001", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100", "40", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 12],
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.le,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.le,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
    ],
)
@pytest.mark.parametrize("integer_dtype", utils.INTEGER_TYPES)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_comp_mixed_integer(args, integer_dtype, reflected):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 decimal data setups, with scale from {==0, >0, <0}.
    Decimal precisions are sufficient to hold the digits.
    For each decimal data setup, there is at least one row that lead to one
    of the following compare results: {True, False, None}.
    """
    if not reflected:
        op, ldata, ldtype, rdata, expected, _ = args
    else:
        op, ldata, ldtype, rdata, _, expected = args

    lhs = _decimal_series(ldata, ldtype)
    rhs = cudf.Series(rdata, dtype=integer_dtype)

    if reflected:
        rhs, lhs = lhs, rhs

    if integer_dtype in {'int64', 'uint64'}:
        with pytest.raises(TypeError):
            op(lhs, rhs)
        return

    actual = op(lhs, rhs)
    utils.assert_eq(expected, actual)



@pytest.mark.parametrize("args", [
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int8'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(4, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int8'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(5, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int8'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(4, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int16'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(6, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int16'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(7, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int16'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(6, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int32'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(11, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int32'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(12, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int32'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(11, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint8'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(4, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint8'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(5, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint8'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(4, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint16'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(6, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint16'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(7, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint16'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(6, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint32'),
    operator.add,
    ['2', '4', '6'],
    cudf.Decimal64Dtype(11, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint32'),
    operator.mul,
    ['1', '4', '9'],
    cudf.Decimal64Dtype(12, 0),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint32'),
    operator.sub,
    ['0', '0', '0'],
    cudf.Decimal64Dtype(11, 0),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int8'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(5, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int8'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(6, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int8'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(5, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int16'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(7, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int16'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(8, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int16'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(7, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int32'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(12, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int32'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(13, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('int32'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(12, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint8'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(5, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint8'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(6, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint8'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(5, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint16'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(7, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint16'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(8, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint16'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(7, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint32'),
    operator.add,
    ['2.1', '4.1', '6.1'],
    cudf.Decimal64Dtype(12, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint32'),
    operator.mul,
    ['1.1', '4.2', '9.3'],
    cudf.Decimal64Dtype(13, 1),
    ),
    (
    ['1.1', '2.1', '3.1'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(2, 1),
    np.dtype('uint32'),
    operator.sub,
    ['0.1', '0.1', '0.1'],
    cudf.Decimal64Dtype(12, 1),
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint64'),
    operator.add,
    TypeError,
    None,
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint64'),
    operator.sub,
    TypeError,
    None,
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('uint64'),
    operator.mul,
    TypeError,
    None,
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int64'),
    operator.add,
    TypeError,
    None,
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int64'),
    operator.sub,
    TypeError,
    None,
    ),
    (
    ['1', '2', '3'],
    ['1', '2', '3'],
    cudf.Decimal64Dtype(1, 0),
    np.dtype('int64'),
    operator.mul,
    TypeError,
    None,
    ),
])
def test_binops_decimal_integer_column(args):

    ldata, rdata, ldtype, rdtype, op, expect, expect_dtype = args
    lhs = _decimal_series(ldata, ldtype)
    rhs = cudf.core.column.as_column(rdata, dtype=rdtype)

    if expect == TypeError:
        with pytest.raises(expect):
            op(lhs, rhs)
    else:
        expect = _decimal_series(expect, dtype=expect_dtype)
        got = op(lhs, rhs)
        utils.assert_eq(expect, got)

@pytest.mark.parametrize("args", [
    # Addition, non reflected
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.add,
        np.dtype('float64'),
        False
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.add,
        np.dtype('float32'),
        False
    ),
    # Addition, reflected    
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.add,
        np.dtype('float64'),
        True
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.add,
        np.dtype('float32'),
        True
    ),
    # Subtraction, float from decimal
    (
        ['1.1', '2.2', '3.3'],
        [1.05, 2.05, 3.05],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.sub,
        np.dtype('float64'),
        False
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.05, 2.05, 3.05],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.sub,
        np.dtype('float32'),
        False
    ),
    # Subtraction, decimal from float
    (
        ['1.05', '2.05', '3.05'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(3,2),
        np.dtype('float64'),
        operator.sub,
        np.dtype('float64'),
        True
    ),
    (
        ['1.05', '2.05', '3.05'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(3,2),
        np.dtype('float32'),
        operator.sub,
        np.dtype('float32'),
        True
    ),
    # Multiplication, non-reflected
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.mul,
        np.dtype('float64'),
        False
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.mul,
        np.dtype('float32'),
        False
    ),
    # Multiplication, reflected
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.mul,
        np.dtype('float64'),
        True
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.mul,
        np.dtype('float32'),
        True
    ),     
    # division, float by decimal
    (
        ['1.1', '2.2', '3.3'],
        [1.05, 2.05, 3.05],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float64'),
        operator.truediv,
        np.dtype('float64'),
        False
    ),
    (
        ['1.1', '2.2', '3.3'],
        [1.05, 2.05, 3.05],
        cudf.Decimal64Dtype(2,1),
        np.dtype('float32'),
        operator.truediv,
        np.dtype('float32'),
        False
    ),
    # Subtraction, decimal from float
    (
        ['1.05', '2.05', '3.05'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(3,2),
        np.dtype('float64'),
        operator.truediv,
        np.dtype('float64'),
        True
    ),
    (
        ['1.05', '2.05', '3.05'],
        [1.1, 2.2, 3.3],
        cudf.Decimal64Dtype(3,2),
        np.dtype('float32'),
        operator.truediv,
        np.dtype('float32'),
        True
    ),   
])

def test_binops_decimal_float_column(args):
    ldata, rdata, ldtype, rdtype, op, expect_dtype, reflect = args

    lhs = _decimal_series(ldata, ldtype)
    rhs = cudf.Series(rdata, dtype=rdtype)

    def make_expect(lhs, rhs, ldtype, rdtype):
        ans = []
        for lhs, rhs in zip(ldata, rdata):
            # ultimately we will be operating on two floats
            # the answer we want though is the same thing that
            # we would get if we cast a decimal.Decimal to 
            # a float of the correct width and then used that
            # in the operation rather than the decimal. 

            # ldata: element->decimal->f32/64->pythonfloat
            # example: '0.05'
            # for float32 -> 0.05000000074505805969...
            # for float64 -> 0.05000000000000000277...
            lhs = float(
                rdtype.type(
                    decimal.Decimal(
                        lhs
                    )
                )
            )
            rhs = rdtype.type(rhs)
            if reflect:
                lhs, rhs = rhs, lhs
            ans.append(op(lhs, rhs))
        return ans

    expect_data = make_expect(lhs, rhs, ldtype, rdtype)
    expect = cudf.Series(expect_data, dtype=expect_dtype)
    if reflect:
        lhs, rhs = rhs, lhs

    # result will be float, not decimal for these binops
    breakpoint()
    got = op(lhs, rhs)
    utils.assert_eq(expect, got)



@pytest.mark.parametrize(
    "args",
    [
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("1.5")),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["98", "198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["97.5", "197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["96", "196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("2.5")),
            ["97.5", "197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["-98", "-198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["-96", "-196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            cudf.Scalar(decimal.Decimal("2.5")),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
    ],
)
def test_binops_decimal_scalar(args):
    op, lhs, l_dtype, rhs, expect, expect_dtype, reflect = args

    def decimal_series(input, dtype):
        return cudf.Series(
            [x if x is None else decimal.Decimal(x) for x in input],
            dtype=dtype,
        )

    lhs = decimal_series(lhs, l_dtype)
    expect = decimal_series(expect, expect_dtype)

    if reflect:
        lhs, rhs = rhs, lhs

    got = op(lhs, rhs)
    assert expect.dtype == got.dtype
    utils.assert_eq(expect, got)


@pytest.mark.parametrize(
    "args",
    [
        (
            operator.eq,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            100,
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, False, None], dtype=bool),
            cudf.Series([True, False, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([False, False, True, None], dtype=bool),
            cudf.Series([False, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, False, True, None], dtype=bool),
            cudf.Series([True, True, False, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([False, True, False, None], dtype=bool),
            cudf.Series([False, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            cudf.Scalar(decimal.Decimal("100.123")),
            cudf.Series([True, True, False, None], dtype=bool),
            cudf.Series([True, False, True, None], dtype=bool),
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_scalar_compare(args, reflected):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 data setups: pyints, Decimal, and
    decimal cudf.Scalar
    For each data setup, there is at least one row that lead to one of the
    following compare results: {True, False, None}.
    """
    if not reflected:
        op, ldata, ldtype, rdata, expected, _ = args
    else:
        op, ldata, ldtype, rdata, _, expected = args

    lhs = _decimal_series(ldata, ldtype)
    rhs = rdata

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    utils.assert_eq(expected, actual)
