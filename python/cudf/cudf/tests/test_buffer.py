# Copyright (c) 2020-2022, NVIDIA CORPORATION.
import cupy as cp
import pytest

from cudf.core.buffer import buffer_from_pointer

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
        buffer_from_pointer(ptr=data.view("|u1"), size=data.size, owner=None)
    else:
        with pytest.raises(ValueError):
            buffer_from_pointer(
                ptr=data.view("|u1"), size=data.size, owner=None
            )


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
    buffer_from_pointer(ptr=data, size=data.size, owner=None)
