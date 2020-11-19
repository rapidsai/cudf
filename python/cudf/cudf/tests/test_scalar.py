import datetime
import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import Scalar as pycudf_scalar
from cudf._lib.copying import get_element
from cudf.tests.utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)

SCALAR_VALUES = [
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
    np.timedelta64(0, "s"),
    np.timedelta64(1, "s"),
    np.timedelta64(-1, "s"),
    np.timedelta64(42, "s"),
    np.timedelta64(np.iinfo(np.int64).max, "s"),
    np.timedelta64(np.iinfo(np.int64).min + 1, "s"),
    np.timedelta64(42, "ms"),
    np.timedelta64(np.iinfo(np.int64).max, "ms"),
    np.timedelta64(np.iinfo(np.int64).min + 1, "ms"),
    np.timedelta64(42, "us"),
    np.timedelta64(np.iinfo(np.int64).max, "us"),
    np.timedelta64(np.iinfo(np.int64).min + 1, "us"),
    np.timedelta64(42, "ns"),
    np.timedelta64(np.iinfo(np.int64).max, "ns"),
    np.timedelta64(np.iinfo(np.int64).min + 1, "ns"),
    "",
    "one",
    "1",
    True,
    False,
    np.bool_(True),
    np.bool_(False),
    np.str_("asdf"),
    np.object_("asdf"),
]


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_scalar_host_initialization(value):
    s = cudf.Scalar(value)

    np.testing.assert_equal(s.value, value)
    assert s.is_valid() is True
    assert s._is_host_value_current
    assert not s._is_device_value_current


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_scalar_device_initialization(value):
    column = cudf.Series([value], nan_as_null=False)._column
    dev_slr = get_element(column, 0)

    s = cudf.Scalar(dev_slr)

    assert s._is_device_value_current
    assert not s._is_host_value_current

    assert s.value == value or np.isnan(s.value) and np.isnan(value)

    assert s._is_device_value_current
    assert s._is_host_value_current


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_scalar_roundtrip(value):
    s = cudf.Scalar(value)

    assert s._is_host_value_current
    assert not s._is_device_value_current

    # call this property to sync the scalar
    s.device_value

    assert s._is_host_value_current
    assert s._is_device_value_current

    # invalidate the host cache
    s._host_value = None
    s._host_dtype = None

    assert not s._is_host_value_current
    assert s._is_device_value_current

    # this should trigger a host copy

    assert s.value == value or np.isnan(s.value) and np.isnan(value)


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["object"]
)
def test_null_scalar(dtype):
    s = cudf.Scalar(None, dtype=dtype)
    assert s.value is cudf.NA
    assert s.dtype == np.dtype(dtype)
    assert s.is_valid() is False


@pytest.mark.parametrize(
    "value",
    [
        np.datetime64("NaT", "ns"),
        np.datetime64("NaT", "us"),
        np.datetime64("NaT", "ms"),
        np.datetime64("NaT", "s"),
        np.timedelta64("NaT", "ns"),
        np.timedelta64("NaT", "us"),
        np.timedelta64("NaT", "ms"),
        np.timedelta64("NaT", "s"),
    ],
)
def test_nat_to_null_scalar_succeeds(value):
    s = cudf.Scalar(value)
    assert s.value is cudf.NA
    assert not s.is_valid()
    assert s.dtype == value.dtype


@pytest.mark.parametrize(
    "value", [None, np.datetime64("NaT"), np.timedelta64("NaT")]
)
def test_generic_null_scalar_construction_fails(value):
    with pytest.raises(TypeError):
        cudf.Scalar(value)


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["object"]
)
def test_scalar_dtype_and_validity(dtype):
    s = cudf.Scalar(1, dtype=dtype)

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
    s = cudf.Scalar(value)

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


def test_scalar_implicit_bool_conversion():
    assert cudf.Scalar(True)
    assert not cudf.Scalar(False)
    assert cudf.Scalar(0) == cudf.Scalar(0)
    assert cudf.Scalar(1) <= cudf.Scalar(2)
    assert cudf.Scalar(1) <= 2


@pytest.mark.parametrize("value", [1, -1, 1.5, 0, "1.5", "1", True, False])
def test_scalar_implicit_float_conversion(value):
    expect = float(value)
    got = float(cudf.Scalar(value))

    assert expect == got
    assert type(expect) == type(got)


@pytest.mark.parametrize("value", [1, -1, 1.5, 0, "1", True, False])
def test_scalar_implicit_int_conversion(value):
    expect = int(value)
    got = int(cudf.Scalar(value))

    assert expect == got
    assert type(expect) == type(got)


@pytest.mark.parametrize("cls", [int, float, bool])
@pytest.mark.parametrize("dtype", sorted(set(ALL_TYPES) - {"category"}))
def test_scalar_invalid_implicit_conversion(cls, dtype):

    try:
        cls(pd.NA)
    except TypeError as e:

        error = (
            str(e).replace("NAType", "_NAType").replace(" NA ", " cudf.NA ")
        )
        with pytest.raises(TypeError, match=re.escape(str(error))):
            slr = pycudf_scalar(None, dtype=dtype)
            cls(slr)


@pytest.mark.parametrize("value", SCALAR_VALUES)
def test_device_scalar_direct_construction(value):
    value = cudf.utils.utils.to_cudf_compatible_scalar(value)
    dtype = value.dtype

    s = cudf._lib.scalar.DeviceScalar(value, dtype)

    assert s.value == value or np.isnan(s.value) and np.isnan(value)
    if dtype.char == "U":
        assert s.dtype == "object"
    else:
        assert s.dtype == dtype
