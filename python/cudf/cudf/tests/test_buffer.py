# Copyright (c) 2020-2022, NVIDIA CORPORATION.
from typing import Callable

import cupy as cp
import pytest

from cudf.core.buffer import Buffer, DeviceBufferLike, as_device_buffer_like

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
        as_device_buffer_like(data.view("|u1"))
    else:
        with pytest.raises(ValueError):
            as_device_buffer_like(data.view("|u1"))


@pytest.mark.parametrize(
    "data",
    [
        cp.arange(arr_len),
        cp.arange(arr_len).reshape(1, arr_len),
        cp.arange(arr_len).reshape(1, arr_len, 1),
        cp.arange(arr_len**2).reshape(arr_len, arr_len),
    ],
)
@pytest.mark.parametrize("dtype", ["uint8", "int8", "float32", "int32"])
def test_buffer_from_cuda_iface_dtype(data, dtype):
    data = data.astype(dtype)
    buf = as_device_buffer_like(data)
    ary = cp.array(buf).flatten().view("uint8")
    assert (ary == buf).all()


@pytest.mark.parametrize("creator", [Buffer, as_device_buffer_like])
def test_buffer_creation_from_any(creator: Callable[[object], Buffer]):
    ary = cp.arange(arr_len)
    b = creator(ary)
    assert isinstance(b, DeviceBufferLike)
    assert ary.__cuda_array_interface__["data"][0] == b.ptr
    assert ary.nbytes == b.size

    with pytest.raises(
        ValueError, match="size must be specified when `data` is an integer"
    ):
        Buffer(42)


@pytest.mark.parametrize(
    "size,expect", [(10, "10B"), (2**10 + 500, "1.49KiB"), (2**20, "1MiB")]
)
def test_buffer_repr(size, expect):
    ary = cp.arange(size, dtype="uint8")
    buf = as_device_buffer_like(ary)
    assert f"size={expect}" in repr(buf)


@pytest.mark.parametrize(
    "idx",
    [
        slice(0, 0),
        slice(0, 1),
        slice(-2, -1),
        slice(0, arr_len),
        slice(2, 3),
        slice(2, -1),
    ],
)
def test_buffer_slice(idx):
    ary = cp.arange(arr_len, dtype="uint8")
    buf = as_device_buffer_like(ary)
    assert (ary[idx] == buf[idx]).all()


@pytest.mark.parametrize(
    "idx, err_msg",
    [
        (1, "index must be an slice"),
        (slice(3, 2), "size cannot be negative"),
        (slice(1, 2, 2), "slice must be contiguous"),
        (slice(1, 2, -1), "slice must be contiguous"),
        (slice(3, 2, -1), "slice must be contiguous"),
    ],
)
def test_buffer_slice_fail(idx, err_msg):
    ary = cp.arange(arr_len, dtype="uint8")
    buf = as_device_buffer_like(ary)

    with pytest.raises(ValueError, match=err_msg):
        buf[idx]
