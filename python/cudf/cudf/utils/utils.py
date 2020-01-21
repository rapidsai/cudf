import functools
from math import ceil, floor, isinf, isnan

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import njit

import rmm

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
def rint(x):
    """Round to the nearest integer.

    Returns
    -------
    The nearest integer, as a float.
    """
    y = floor(x)
    r = x - y

    if r > 0.5:
        y += 1.0
    if r == 0.5:
        r = y - 2.0 * floor(0.5 * y)
        if r == 1.0:
            y += 1.0
    return y


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


def scalar_broadcast_to(scalar, size, dtype=None):
    from cudf.utils.cudautils import fill_value
    from cudf.utils.dtypes import to_cudf_compatible_scalar, is_string_dtype
    from cudf.core.column import column_empty

    if isinstance(size, (tuple, list)):
        size = size[0]

    if scalar is None:
        if dtype is None:
            dtype = "object"
        return column_empty(size, dtype=dtype, masked=True)

    if isinstance(scalar, pd.Categorical):
        return scalar_broadcast_to(scalar.categories[0], size).astype(dtype)

    if isinstance(scalar, str) and (is_string_dtype(dtype) or dtype is None):
        dtype = "object"
    else:
        scalar = to_cudf_compatible_scalar(scalar, dtype=dtype)
        dtype = scalar.dtype

    if np.dtype(dtype) == np.dtype("object"):
        import nvstrings
        from cudf.core.column import as_column
        from cudf.utils.cudautils import zeros

        gather_map = zeros(size, dtype="int32")
        scalar_str_col = as_column(nvstrings.to_device([scalar]))
        return scalar_str_col[gather_map]
    else:
        da = rmm.device_array((size,), dtype=dtype)
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

    if pa_arr.null_count:
        mask_dev_array = make_mask(len(pa_arr))
        arrow_dev_array = rmm.to_device(np.asarray(buffers[0]).view("int8"))
        copy_array(arrow_dev_array, mask_dev_array)
        pamask = Buffer(mask_dev_array)
    else:
        pamask = None

    offset = pa_arr.offset
    size = pa_arr.offset + len(pa_arr)

    if dtype:
        data_dtype = dtype
    elif isinstance(pa_arr, pa.StringArray):
        data_dtype = np.int32
        size = size + 1  # extra element holds number of bytes
    else:
        if isinstance(pa_arr, pa.DictionaryArray):
            data_dtype = pa_arr.indices.type.to_pandas_dtype()
        else:
            data_dtype = pa_arr.type.to_pandas_dtype()

    if buffers[1]:
        padata = Buffer(
            np.asarray(buffers[1]).view(data_dtype)[offset : offset + size]
        )
    else:
        padata = Buffer.empty(0)

    pastrs = None
    if isinstance(pa_arr, pa.StringArray):
        pastrs = Buffer(np.asarray(buffers[2]).view(np.int8))
    return (pamask, padata, pastrs)


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


def initfunc(f):
    """
    Decorator for initialization functions that should
    be run exactly once.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if wrapper.initialized:
            return
        wrapper.initialized = True
        return f(*args, **kwargs)

    wrapper.initialized = False
    return wrapper


def get_null_series(size, dtype=np.bool):
    """
    Creates a null series of provided dtype and size

    Parameters
    ----------
    size:  length of series
    dtype: dtype of series to create; defaults to bool.

    Returns
    -------
    a null cudf series of provided `size` and `dtype`
    """
    from cudf.core import Series, column

    empty_col = column.column_empty(size, dtype, True)
    return Series(empty_col)


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


@initfunc
def set_allocator(
    allocator="default",
    pool=False,
    initial_pool_size=None,
    enable_logging=False,
):
    """
    Set the GPU memory allocator. This function should be run only once,
    before any cudf objects are created.

    allocator : {"default", "managed"}
        "default": use default allocator.
        "managed": use managed memory allocator.
    pool : bool
        Enable memory pool.
    initial_pool_size : int
        Memory pool size in bytes. If ``None`` (default), 1/2 of total
        GPU memory is used. If ``pool=False``, this argument is ignored.
    enable_logging : bool, optional
        Enable logging (default ``False``).
        Enabling this option will introduce performance overhead.
    """
    use_managed_memory = True if allocator == "managed" else False

    rmm.reinitialize(
        pool_allocator=pool,
        managed_memory=use_managed_memory,
        initial_pool_size=initial_pool_size,
        logging=enable_logging,
    )


IS_NEP18_ACTIVE = _is_nep18_active()


class cached_property:
    """
    Like @property, but only evaluated upon first invocation.
    To force re-evaluation of a cached_property, simply delete
    it with `del`.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
