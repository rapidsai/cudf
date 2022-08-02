# Copyright (c) 2020-2022, NVIDIA CORPORATION.
import cupy as cp
import pytest
from cupy.testing import assert_array_equal

from cudf.core.buffer import Buffer, as_buffer

arr_len = 10


@pytest.mark.parametrize(
    "data",
    [
        (cp.zeros(arr_len), True),
        (cp.zeros((1, arr_len)), True),
        (cp.zeros((1, arr_len, 1)), True),
        (cp.zeros((arr_len, arr_len)), True),
        (cp.zeros((arr_len, arr_len)).reshape(arr_len * arr_len), True),
        (cp.zeros((arr_len, arr_len))[:, 0], False),
    ],
)
def test_buffer_from_cuda_iface_contiguous(data):
    data, expect_success = data
    if expect_success:
        buf = Buffer(data=data.view("|u1"), size=data.size)  # noqa: F841
    else:
        with pytest.raises(ValueError):
            buf = Buffer(data=data.view("|u1"), size=data.size)  # noqa: F841


@pytest.mark.parametrize(
    "data",
    [
        (cp.zeros(arr_len)),
        (cp.zeros((1, arr_len))),
        (cp.zeros((1, arr_len, 1))),
        (cp.zeros((arr_len, arr_len))),
        (cp.zeros((arr_len, arr_len)).reshape(arr_len * arr_len)),
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "int8", "float32", "int32"])
def test_buffer_from_cuda_iface_dtype(data, dtype):
    data = data.astype(dtype)
    Buffer(data=data, size=data.size)


@pytest.mark.parametrize("size", [0, 1, 10, 100, 1000, 10_000])
def test_buffer_copy(size):
    data = cp.random.randint(low=0, high=100, size=size, dtype="u1")
    buf = as_buffer(data)
    got = buf.copy()
    assert got.size == buf.size
    if size > 0:
        assert got.ptr != buf.ptr
    assert_array_equal(cp.asarray(buf), cp.asarray(got))
