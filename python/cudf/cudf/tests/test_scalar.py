import numpy as np
import pytest

from cudf._lib.scalar import Scalar


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


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "object",
    ],
)
def test_null_scalar(dtype):
    s = Scalar(None, dtype=dtype)
    assert s.value is None
    assert s.dtype == np.dtype(dtype)
