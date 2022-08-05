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
    as_device_buffer_like(data)


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
