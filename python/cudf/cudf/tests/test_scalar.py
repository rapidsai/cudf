# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import datetime
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging import version

import rmm

import cudf
from cudf._lib.copying import get_element
from cudf.testing._utils import (
    ALL_TYPES,
    DATETIME_TYPES,
    NUMERIC_TYPES,
    TIMEDELTA_TYPES,
)


@pytest.fixture(autouse=True)
def clear_scalar_cache():
    cudf.Scalar._clear_instance_cache()
    yield


TEST_DECIMAL_TYPES = [
    cudf.Decimal64Dtype(1, 1),
    cudf.Decimal64Dtype(4, 2),
    cudf.Decimal64Dtype(4, -2),
    cudf.Decimal32Dtype(3, 1),
    cudf.Decimal128Dtype(28, 3),
]

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

DECIMAL_VALUES = [
    Decimal("100"),
    Decimal("0.0042"),
    Decimal("1.0042"),
]


@pytest.mark.parametrize("value", SCALAR_VALUES + DECIMAL_VALUES)
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

    s = cudf.Scalar.from_device_scalar(dev_slr)

    assert s._is_device_value_current
    assert not s._is_host_value_current

    assert s.value == value or np.isnan(s.value) and np.isnan(value)

    assert s._is_device_value_current
    assert s._is_host_value_current


@pytest.mark.parametrize("value", DECIMAL_VALUES)
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_scalar_device_initialization_decimal(value, decimal_type):
    dtype = decimal_type._from_decimal(value)
    column = cudf.Series([str(value)]).astype(dtype)._column
    dev_slr = get_element(column, 0)

    s = cudf.Scalar.from_device_scalar(dev_slr)

    assert s._is_device_value_current
    assert not s._is_host_value_current

    assert s.value == value

    assert s._is_device_value_current
    assert s._is_host_value_current


@pytest.mark.parametrize("value", SCALAR_VALUES + DECIMAL_VALUES)
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
    "dtype",
    NUMERIC_TYPES
    + DATETIME_TYPES
    + TIMEDELTA_TYPES
    + ["object"]
    + TEST_DECIMAL_TYPES,
)
def test_null_scalar(dtype):
    s = cudf.Scalar(None, dtype=dtype)
    if s.dtype.kind in "mM":
        assert s.value is cudf.NaT
    else:
        assert s.value is cudf.NA
    assert s.dtype == (
        cudf.dtype(dtype)
        if not isinstance(dtype, cudf.core.dtypes.DecimalDtype)
        else dtype
    )
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
    assert s.value is cudf.NaT
    assert not s.is_valid()
    assert s.dtype == value.dtype


@pytest.mark.parametrize(
    "value", [None, np.datetime64("NaT"), np.timedelta64("NaT")]
)
def test_generic_null_scalar_construction_fails(value):
    with pytest.raises(TypeError):
        cudf.Scalar(value)


@pytest.mark.parametrize(
    "value, dtype", [(1000, "uint8"), (2**30, "int16"), (-1, "uint16")]
)
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_scalar_out_of_bounds_pyint_fails(value, dtype):
    # Test that we align with NumPy on scalar creation behavior from
    # Python integers.
    if version.parse(np.__version__) >= version.parse("2.0"):
        with pytest.raises(OverflowError):
            cudf.Scalar(value, dtype)
    else:
        # NumPy allowed this, but it gives a DeprecationWarning on newer
        # versions (which cudf did not used to do).
        assert cudf.Scalar(value, dtype).value == np.dtype(dtype).type(value)


@pytest.mark.parametrize(
    "dtype", NUMERIC_TYPES + DATETIME_TYPES + TIMEDELTA_TYPES + ["object"]
)
def test_scalar_dtype_and_validity(dtype):
    s = cudf.Scalar(1, dtype=dtype)

    assert s.dtype == cudf.dtype(dtype)
    assert s.is_valid() is True


@pytest.mark.parametrize(
    "slr,dtype,expect",
    [
        (1, cudf.Decimal64Dtype(1, 0), Decimal("1")),
        (Decimal(1), cudf.Decimal64Dtype(1, 0), Decimal("1")),
        (Decimal("1.1"), cudf.Decimal64Dtype(2, 1), Decimal("1.1")),
        (Decimal("1.1"), cudf.Decimal64Dtype(4, 3), Decimal("1.100")),
        (Decimal("41.123"), cudf.Decimal32Dtype(5, 3), Decimal("41.123")),
        (
            Decimal("41345435344353535344373628492731234.123"),
            cudf.Decimal128Dtype(38, 3),
            Decimal("41345435344353535344373628492731234.123"),
        ),
        (Decimal("1.11"), cudf.Decimal64Dtype(2, 2), pa.lib.ArrowInvalid),
    ],
)
def test_scalar_dtype_and_validity_decimal(slr, dtype, expect):
    if expect is pa.lib.ArrowInvalid:
        with pytest.raises(expect):
            cudf.Scalar(slr, dtype=dtype)
        return
    else:
        result = cudf.Scalar(slr, dtype=dtype)
        assert result.dtype == dtype
        assert result.is_valid


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

    if isinstance(value, datetime.datetime):
        expected = np.datetime64(value)
    elif isinstance(value, datetime.timedelta):
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
    assert type(expect) is type(got)


@pytest.mark.parametrize("value", [1, -1, 1.5, 0, "1", True, False])
def test_scalar_implicit_int_conversion(value):
    expect = int(value)
    got = int(cudf.Scalar(value))

    assert expect == got
    assert type(expect) is type(got)


@pytest.mark.parametrize("cls", [int, float, bool])
@pytest.mark.parametrize("dtype", sorted(set(ALL_TYPES) - {"category"}))
def test_scalar_invalid_implicit_conversion(cls, dtype):
    try:
        cls(pd.NaT if cudf.dtype(dtype).kind in "mM" else pd.NA)
    except TypeError as e:
        with pytest.raises(TypeError, match=re.escape(str(e))):
            slr = cudf.Scalar(None, dtype=dtype)
            cls(slr)


@pytest.mark.parametrize("value", SCALAR_VALUES + DECIMAL_VALUES)
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_device_scalar_direct_construction(value, decimal_type):
    value = cudf.utils.dtypes.to_cudf_compatible_scalar(value)

    dtype = (
        value.dtype
        if not isinstance(value, Decimal)
        else decimal_type._from_decimal(value)
    )

    s = cudf._lib.scalar.DeviceScalar(value, dtype)

    assert s.value == value or np.isnan(s.value) and np.isnan(value)
    if isinstance(
        dtype, (cudf.Decimal64Dtype, cudf.Decimal128Dtype, cudf.Decimal32Dtype)
    ):
        assert s.dtype.precision == dtype.precision
        assert s.dtype.scale == dtype.scale
    elif dtype.char == "U":
        assert s.dtype == "object"
    else:
        assert s.dtype == dtype


@pytest.mark.parametrize("value", SCALAR_VALUES + DECIMAL_VALUES)
def test_construct_from_scalar(value):
    value = cudf.utils.dtypes.to_cudf_compatible_scalar(value)
    x = cudf.Scalar(
        value, value.dtype if not isinstance(value, Decimal) else None
    )
    y = cudf.Scalar(x)
    assert x.value == y.value or np.isnan(x.value) and np.isnan(y.value)

    # check that this works:
    y.device_value

    x._is_host_value_current == y._is_host_value_current
    x._is_device_value_current == y._is_device_value_current


@pytest.mark.parametrize(
    "data", ["20000101", "2000-01-01", "2000-01-01T00:00:00.000000000", "2000"]
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_datetime_scalar_from_string(data, dtype):
    slr = cudf.Scalar(data, dtype)

    expected = np.datetime64(datetime.datetime(2000, 1, 1)).astype(dtype)

    assert expected == slr.value


def test_scalar_cache():
    s = cudf.Scalar(1)
    s2 = cudf.Scalar(1)

    assert s is s2


def test_scalar_cache_rmm_hook():
    # test that reinitializing rmm clears the cuDF scalar cache, as we
    # register a hook with RMM that does that on reinitialization
    s = cudf.Scalar(1)
    s2 = cudf.Scalar(1)

    assert s is s2

    rmm.reinitialize()

    s3 = cudf.Scalar(1)
    assert s3 is not s


def test_default_integer_bitwidth_scalar(default_integer_bitwidth):
    # Test that integer scalars are default to 32 bits under user options.
    slr = cudf.Scalar(128)
    assert slr.dtype == np.dtype(f"i{default_integer_bitwidth//8}")


def test_default_float_bitwidth_scalar(default_float_bitwidth):
    # Test that float scalars are default to 32 bits under user options.
    slr = cudf.Scalar(128.0)
    assert slr.dtype == np.dtype(f"f{default_float_bitwidth//8}")


def test_scalar_numpy_casting():
    # binop should upcast to wider type
    s1 = cudf.Scalar(1, dtype=np.int32)
    s2 = np.int64(2)
    assert s1 < s2


def test_construct_timezone_scalar_error():
    pd_scalar = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        cudf.utils.dtypes.to_cudf_compatible_scalar(pd_scalar)

    date_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        cudf.utils.dtypes.to_cudf_compatible_scalar(date_scalar)
