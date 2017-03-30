from itertools import product

import numpy as np

from numba import cuda, vectorize, types


def to_device(ary):
    dary, _ = cuda._auto_device(ary)
    return dary


# GPU array type casting


@cuda.jit
def gpu_copy(inp, out):
    tid = cuda.grid(1)
    if tid < out.size:
        out[tid] = inp[tid]


def astype(ary, dtype):
    if ary.dtype == np.dtype(dtype):
        return ary
    else:
        out = cuda.device_array(shape=ary.shape, dtype=dtype)
        configured = gpu_copy.forall(out.size)
        configured(ary, out)
        return out


# Copy column into a matrix

@cuda.jit
def gpu_copy_column(matrix, colidx, colvals):
    tid = cuda.grid(1)
    if tid < colvals.size:
        matrix[colidx, tid] = colvals[tid]


def copy_column(matrix, colidx, colvals):
    assert matrix.shape[1] == colvals.size
    configured = gpu_copy_column.forall(colvals.size)
    configured(matrix, colidx, colvals)
