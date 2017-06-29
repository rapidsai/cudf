from collections import namedtuple

import numpy as np

from numba import njit, cuda

mask_dtype = np.dtype(np.uint8)
mask_bitsize = mask_dtype.itemsize * 8


def calc_chunk_size(size, chunksize):
    return (size + chunksize - 1) // chunksize


_TypeMinMax = namedtuple('_TypeMinMax', 'min,max')


def get_numeric_type_info(dtype):
    if dtype.kind in 'iu':
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind in 'f':
        return _TypeMinMax(dtype.type('-inf'), dtype.type('+inf'))
    else:
        raise TypeError(dtype)


@njit
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


@njit
def mask_set(mask, pos):
    mask[pos // mask_bitsize] |= 1 << (pos % mask_bitsize)


def boolmask_to_bitmask(bools):
    masksize = calc_chunk_size(bools.size, mask_bitsize)
    mask = np.zeros(masksize, dtype=mask_dtype)
    for i, x in enumerate(bools):
        if x:
            mask_set(mask, i)
    return mask


def make_mask(size):
    """Create mask to obtain at least *size* number of bits.
    """
    size = calc_chunk_size(size, mask_bitsize)
    return cuda.device_array(shape=size, dtype=mask_dtype)


def scalar_broadcast_to(scalar, shape):
    if not isinstance(shape, tuple):
        shape = (shape,)
    arr = np.broadcast_to(np.asarray(scalar), shape=shape)
    # FIXME: this is wasteful, but numba can't slice 0-strided array
    arr = np.ascontiguousarray(arr)
    return cuda.to_device(arr)


def normalize_index(index, size, doraise=True):
    """Normalize negative index
    """
    if index < 0:
        index = size + index
    if doraise and not (0 <= index < size):
        raise IndexError('out-of-bound')
    return min(index, size)

