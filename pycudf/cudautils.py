from itertools import product

import numpy as np

from numba import cuda, vectorize, types, uint64

from .utils import mask_bitsize


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


# Mask utils

@cuda.jit
def gpu_set_mask_from_stride(mask, stride):
    bitsize = mask_bitsize
    tid = cuda.grid(1)
    if tid < mask.size:
        base = tid * bitsize
        val = 0
        for shift in range(bitsize):
            idx = base + shift
            bit = (idx % stride) == 0
            val |= bit << shift
        mask[tid] = val


def set_mask_from_stride(mask, stride):
    taskct = mask.size
    configured = gpu_set_mask_from_stride.forall(taskct)
    configured(mask, stride)


@cuda.jit(device=True)
def gpu_mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


@cuda.jit(device=True)
def gpu_prefixsum(data, initial):
    # XXX slow imp
    acc = initial
    for i in range(data.size):
        tmp = data[i]
        data[i] = acc
        acc += tmp
    return acc


@cuda.jit
def gpu_mask_assign_slot(mask, slots):
    """
    Assign output slot from the given mask.

    Note:
    * this is a single wrap, single block kernel
    """
    tid = cuda.threadIdx.x
    blksz = cuda.blockDim.x

    sm_prefix = cuda.shared.array(shape=(33,), dtype=uint64)
    offset = 0
    for base in range(0, slots.size, blksz):
        i = base + tid
        sm_prefix[tid] = 0
        if i < slots.size:
            sm_prefix[tid] = gpu_mask_get(mask, i)
        if tid == 0:
            offset = gpu_prefixsum(sm_prefix[:32], offset)
            sm_prefix[32] = offset
        offset = sm_prefix[32]
        if i < slots.size:
            slots[i] = sm_prefix[tid]


@cuda.jit
def gpu_copy_to_dense(data, mask, slots, out):
    tid = cuda.grid(1)
    if tid < data.size and gpu_mask_get(mask, tid):
        idx = slots[tid]
        out[idx] = data[tid]


def copy_to_dense(data, mask, out=None):
    """
    The output array can be specified in `out`.

    Return a 2-tuple of:
    * number of non-null element
    * a dense gpu array given the data and mask gpu arrays.
    """
    slots = cuda.device_array(shape=data.shape, dtype=np.uint64)
    gpu_mask_assign_slot[1, 32](mask, slots)
    sz = slots[-1:].copy_to_host() + 1
    if out is None:
        out = cuda.device_array(shape=sz, dtype=data.dtype)
    else:
        # check
        if sz >= out.size:
            raise ValueError('output array too small')
    gpu_copy_to_dense.forall(data.size)(data, mask, slots, out)
    return (sz, out)


#
# Fill NA
#


@cuda.jit
def gpu_fill_masked(value, validity, out):
    tid = cuda.grid(1)
    if tid < out.size:
        valid = gpu_mask_get(validity, tid)
        if not valid:
            out[tid] = value


def fillna(data, mask, value):
    out = cuda.device_array_like(data)
    out.copy_to_device(data)
    configured = gpu_fill_masked.forall(data.size)
    configured(value, mask, out)
    return out

#
# Binary kernels
#

@cuda.jit
def gpu_equal_constant(arr, val, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (arr[i] == val)


def apply_equal_constant(arr, val, dtype):
    out = cuda.device_array(shape=arr.size, dtype=dtype)
    configured = gpu_equal_constant.forall(out.size)
    configured(arr, val, out)
    return out

