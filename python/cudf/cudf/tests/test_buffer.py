from cudf.core.buffer import Buffer
import cupy as cp
import numba

import pytest

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
def test_buffer_from_cuda_array_interface(data):
    data, expect_success = data
    if expect_success:
        buf = Buffer(data=data, size=data.size)
    else:
        with pytest.raises(ValueError):
            buf = Buffer(data=data, size=data.size)
