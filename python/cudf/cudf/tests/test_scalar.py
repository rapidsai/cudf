import numpy as np
import pytest

from cudf._libxx.scalar import Scalar


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
        np.int16(1),
        np.int32(42),
        np.int32(-42),
        np.int64(42),
        np.float32(1),
        np.float32(-1),
        np.float64(1),
        np.float64(-1),
        np.datetime64(0, "s"),
        np.datetime64(1, "s"),
        np.datetime64(-1, "s"),
        np.datetime64(42, "s"),
        np.datetime64(42, "ms"),
        np.datetime64(42, "us"),
        np.datetime64(42, "ns"),
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
    assert s.value == value


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
