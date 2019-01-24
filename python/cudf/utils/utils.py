from collections import namedtuple

import numpy as np

from numba import njit

from librmm_cffi import librmm as rmm

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


def make_mask(size):
    """Create mask to obtain at least *size* number of bits.
    """
    size = calc_chunk_size(size, mask_bitsize)
    return rmm.device_array(shape=size, dtype=mask_dtype)


def require_writeable_array(arr):
    # This should be fixed in numba (numba issue #2521)
    return np.require(arr, requirements='W')


def scalar_broadcast_to(scalar, shape, dtype):
    from .cudautils import fill_value

    if not isinstance(shape, tuple):
        shape = (shape,)
    da = rmm.device_array(shape, dtype=dtype)
    if da.size != 0:
        fill_value(da, scalar)
    return da


def normalize_index(index, size, doraise=True):
    """Normalize negative index
    """
    if index < 0:
        index = size + index
    if doraise and not (0 <= index < size):
        raise IndexError('out-of-bound')
    return min(index, size)


def normalize_slice(arg, size):
    """Normalize slice
    """
    start = arg.start if arg.start is not None else 0
    stop = arg.stop if arg.stop is not None else size
    return (normalize_index(start, size, doraise=False),
            normalize_index(stop, size, doraise=False))


# borrowed from a wonderful blog:
# https://avilpage.com/2015/03/a-slice-of-python-intelligence-behind.html
def standard_python_slice(len_idx, arg):
    """ Figuring out the missing parameters of slice"""

    start = arg.start
    stop = arg.stop
    step = arg.step

    if step is None:
        step = 1
    if step == 0:
        raise Exception("Step cannot be zero.")

    if start is None:
        start = 0 if step > 0 else len_idx - 1
    else:
        if start < 0:
            start += len_idx
        if start < 0:
            start = 0 if step > 0 else -1
        if start >= len_idx:
            start = len_idx if step > 0 else len_idx - 1

    if stop is None:
        stop = len_idx if step > 0 else -1
    else:
        if stop < 0:
            stop += len_idx
        if stop < 0:
            stop = 0 if step > 0 else -1
        if stop >= len_idx:
            stop = len_idx if step > 0 else len_idx - 1

    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
        slice_length = 0
    elif step < 0:
        slice_length = (stop - start + 1)/step + 1
    else:
        slice_length = (stop - start - 1)/step + 1

    return start, stop, step, slice_length


list_types_tuple = (list, np.array)
