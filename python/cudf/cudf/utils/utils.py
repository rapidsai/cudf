from math import ceil, isinf, isnan

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import njit

from librmm_cffi import librmm as rmm

mask_dtype = np.dtype(np.int8)
mask_bitsize = mask_dtype.itemsize * 8
mask_byte_padding = 64


def calc_chunk_size(size, chunksize):
    return mask_byte_padding * ceil(
        ((size + chunksize - 1) // chunksize) / mask_byte_padding
    )


@njit
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


@njit
def mask_set(mask, pos):
    mask[pos // mask_bitsize] |= 1 << (pos % mask_bitsize)


@njit
def check_equals_float(a, b):
    return (
        a == b
        or (isnan(a) and isnan(b))
        or ((isinf(a) and a < 0) and (isinf(b) and b < 0))
        or ((isinf(a) and a > 0) and (isinf(b) and b > 0))
    )


@njit
def check_equals_int(a, b):
    return a == b


def make_mask(size):
    """Create mask to obtain at least *size* number of bits.
    """
    size = calc_chunk_size(size, mask_bitsize)
    return rmm.device_array(shape=size, dtype=mask_dtype)


def require_writeable_array(arr):
    # This should be fixed in numba (numba issue #2521)
    return np.require(arr, requirements="W")


def scalar_broadcast_to(scalar, shape, dtype):
    from cudf.utils.cudautils import fill_value
    from cudf.utils.dtypes import to_cudf_compatible_scalar

    scalar = to_cudf_compatible_scalar(scalar, dtype=dtype)

    if not isinstance(shape, tuple):
        shape = (shape,)

    if np.dtype(dtype) == np.dtype("object"):
        import nvstrings
        from cudf.core.column import StringColumn
        from cudf.utils.cudautils import zeros

        gather_map = zeros(shape[0], dtype="int32")
        scalar_str_col = StringColumn(nvstrings.to_device([scalar]))
        return scalar_str_col[gather_map]
    else:
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
        raise IndexError("out-of-bound")
    return min(index, size)


list_types_tuple = (list, np.array)


def buffers_from_pyarrow(pa_arr, dtype=None):
    from cudf.core.buffer import Buffer
    from cudf.utils.cudautils import copy_array

    buffers = pa_arr.buffers()

    if buffers[0]:
        mask_dev_array = make_mask(len(pa_arr))
        arrow_dev_array = rmm.to_device(np.array(buffers[0]).view("int8"))
        copy_array(arrow_dev_array, mask_dev_array)
        pamask = Buffer(mask_dev_array)
    else:
        pamask = None

    if dtype:
        new_dtype = dtype
    else:
        if isinstance(pa_arr, pa.DictionaryArray):
            new_dtype = pa_arr.indices.type.to_pandas_dtype()
        else:
            new_dtype = pa_arr.type.to_pandas_dtype()

    if buffers[1]:
        padata = Buffer(
            np.array(buffers[1]).view(new_dtype)[
                pa_arr.offset : pa_arr.offset + len(pa_arr)
            ]
        )
    else:
        padata = Buffer(np.empty(0, dtype=new_dtype))
    return (pamask, padata)


def get_result_name(left, right):
    """
    This function will give appropriate name for the operations
    involving two Series, Index's or combination of both.

    Parameters
    ----------
    left : {Series, Index}
    right : object

    Returns
    -------
    name : object {string or None}
    """
    from cudf import Series, Index

    if isinstance(right, (Series, Index, pd.Series, pd.Index)):
        name = compare_and_get_name(left, right)
    else:
        name = left.name
    return name


def compare_and_get_name(a, b):
    """
    If both a & b have name attribute, and they are
    same return the common name.
    Else, return either one of the name of a or b,
    whichever is present.

    Parameters
    ----------
    a : object
    b : object

    Returns
    -------
    name : str or None
    """
    a_has = hasattr(a, "name")
    b_has = hasattr(b, "name")

    if a_has and b_has:
        if a.name == b.name:
            return a.name
        else:
            return None
    elif a_has:
        return a.name
    elif b_has:
        return b.name
    return None


# taken from dask array
# https://github.com/dask/dask/blob/master/dask/array/utils.py#L352-L363
def _is_nep18_active():
    class A:
        def __array_function__(self, *args, **kwargs):
            return True

    try:
        return np.concatenate([A()])
    except ValueError:
        return False


IS_NEP18_ACTIVE = _is_nep18_active()


def _cupy_available():
    try:
        import cupy as cp  # noqa: F401

        return True
    except ImportError:
        return False


IS_CUPY_AVAILABLE = _cupy_available()
