import datetime
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cudf import Scalar
from cudf.tests.utils import DATETIME_TYPES, NUMERIC_TYPES, TIMEDELTA_TYPES
import operator

@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        42,
        0.0,
        1.0,
        np.int8(0),
        np.int8(1),
        np.int8(-1),
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max,
        np.int16(1),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
        np.int32(42),
        np.int32(-42),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        np.int64(42),
        np.iinfo(np.int64).min,
        np.iinfo(np.int64).max,
        np.uint8(0),
        np.uint8(1),
        np.uint8(255),
        np.iinfo(np.uint8).min,
        np.iinfo(np.uint8).max,
        np.uint16(1),
        np.iinfo(np.uint16).min,
        np.iinfo(np.uint16).max,
        np.uint32(42),
        np.uint32(4294967254),
        np.iinfo(np.uint32).min,
        np.iinfo(np.uint32).max,
        np.uint64(42),
        np.iinfo(np.uint64).min,
        np.uint64(np.iinfo(np.uint64).max),
        np.float32(1),
        np.float32(-1),
        np.finfo(np.float32).min,
        np.finfo(np.float32).max,
        np.float64(1),
        np.float64(-1),
        np.finfo(np.float64).min,
        np.finfo(np.float64).max,
        np.float32("NaN"),
        np.float64("NaN"),
        np.datetime64(0, "s"),
        np.datetime64(1, "s"),
        np.datetime64(-1, "s"),
        np.datetime64(42, "s"),
        np.datetime64(np.iinfo(np.int64).max, "s"),
        np.datetime64(np.iinfo(np.int64).min + 1, "s"),
        np.datetime64(42, "ms"),
        np.datetime64(np.iinfo(np.int64).max, "ms"),
        np.datetime64(np.iinfo(np.int64).min + 1, "ms"),
        np.datetime64(42, "us"),
        np.datetime64(np.iinfo(np.int64).max, "us"),
        np.datetime64(np.iinfo(np.int64).min + 1, "us"),
        np.datetime64(42, "ns"),
        np.datetime64(np.iinfo(np.int64).max, "ns"),
        np.datetime64(np.iinfo(np.int64).min + 1, "ns"),
        np.datetime64("NaT", "s"),
        np.datetime64("NaT", "ms"),
        np.datetime64("NaT", "us"),
        np.datetime64("NaT", "ns"),
        "",
        "one",
        "1",
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        np.str_("asdf"),
        np.object_("asdf"),
    ],
)
def test_round_trip_scalar(value):
    s = Scalar(value)

    np.testing.assert_equal(s.value, value)
    assert s.is_valid() is True


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["object"]
)
def test_null_scalar(dtype):
    s = Scalar(None, dtype=dtype)
    assert s.value is None
    assert s.dtype == np.dtype(dtype)
    assert s.is_valid() is False


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["object"]
)
def test_valid_scalar(dtype):
    s = Scalar(1, dtype=dtype)

    assert s.dtype == np.dtype(dtype)
    assert s.is_valid() is True


@pytest.mark.parametrize(
    "value",
    [
        datetime.timedelta(seconds=76),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=47),
        datetime.timedelta(hours=4427),
        datetime.timedelta(weeks=7134),
        pd.Timestamp(15133.5, unit="s"),
        pd.Timestamp(15133.5, unit="D"),
        pd.Timedelta(1513393355.5, unit="s"),
        pd.Timedelta(34765, unit="D"),
    ],
)
def test_date_duration_scalars(value):
    s = Scalar(value)

    actual = s.value

    if isinstance(value, dt.datetime):
        expected = np.datetime64(value)
    elif isinstance(value, dt.timedelta):
        expected = np.timedelta64(value)
    elif isinstance(value, pd.Timestamp):
        expected = value.to_datetime64()
    elif isinstance(value, pd.Timedelta):
        expected = value.to_timedelta64()

    np.testing.assert_equal(actual, expected)
    assert s.is_valid() is True

@pytest.mark.parametrize('pairs', [
    (1, 1),
    (1, 1.5),
    (-1.5, 1),
    (1, 'a'),
    ('a', 'b'),
    (1.5, 'a'),
    (1, False),
    (False, True),
    (1.5, False),
    (True, 1.5),
    ('a', False),
])
@pytest.mark.parametrize('dtype_l', [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('float32'),
    np.dtype('float64'),
    np.dtype('bool'),
    np.dtype('str')
])
@pytest.mark.parametrize('dtype_r', [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('float32'),
    np.dtype('float64'),
    np.dtype('bool'),
    np.dtype('str')
])
@pytest.mark.parametrize('op', [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv
])
def test_scalar_binops_value(pairs, dtype_l, dtype_r, op):
    lval, rval = pairs
    if (isinstance(lval, str) and dtype_l != np.dtype('str')) or (isinstance(rval, str) and dtype_r != np.dtype('str')):
        pytest.skip("Invalid scalar/dtype combination")



    import re
    try:
        host_value_l = dtype_l.type(lval)
    except ValueError as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            gpu_value_l = Scalar(lval, dtype=dtype_l)
        return
    try:
        host_value_r = dtype_r.type(rval)
    except ValueError as e:
        with pytest.raises(ValueError, match=re.escape(str(e))):
            gpu_value_r = Scalar(rval, dtype=dtype_r)
        return

    gpu_value_l = Scalar(lval, dtype=dtype_l)
    gpu_value_r = Scalar(rval, dtype=dtype_r)
    try:
        expect = op(host_value_l, host_value_r)
    except TypeError:
        with pytest.raises(TypeError):
            got = op(gpu_value_l, gpu_value_r)
        return
    got = op(gpu_value_l, gpu_value_r)
    assert expect == got.value


@pytest.mark.parametrize('dtype_l', [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('float32'),
    np.dtype('float64'),
])
@pytest.mark.parametrize('dtype_r', [
    np.dtype('uint8'),
    np.dtype('uint16'),
    np.dtype('uint32'),
    np.dtype('uint64'),
    np.dtype('int8'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('float32'),
    np.dtype('float64'),
])
@pytest.mark.parametrize('op', [
    operator.add,
    operator.sub,
    operator.mul,
])
@pytest.mark.parametrize('l_valid', [True, False])
@pytest.mark.parametrize('r_valid', [True, False])
def test_scalar_binops_dtype_and_validity(dtype_l, dtype_r, l_valid, r_valid, op):
    l_value = 0 if l_valid else None
    r_value = 0 if r_valid else None

    expect_dtype = op(dtype_l.type(0), dtype_r.type(0)).dtype

    scalar_l = Scalar(l_value, dtype=dtype_l)
    scalar_r = Scalar(r_value, dtype=dtype_r)

    got = op(scalar_l, scalar_r)

    assert got.dtype == expect_dtype
    assert got.is_valid() == (l_valid and r_valid)
