import functools
from math import ceil, isinf, isnan

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import njit

import rmm
from rmm import rmm_config

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


def _have_cupy():
    try:
        import cupy  # noqa: F401

        _have_cupy = True
    except ModuleNotFoundError:
        _have_cupy = False
    return _have_cupy


def _set_rmm_config(
    use_managed_memory=False,
    use_pool_allocator=False,
    initial_pool_size=None,
    enable_logging=False,
):
    """
    Parameters
    ----------
    use_managed_memory : bool, optional
        If ``True``, use cudaMallocManaged as underlying allocator.
        If ``False`` (default), use  cudaMalloc.
    use_pool_allocator : bool
        If ``True``, enable pool mode.
        If ``False`` (default), disable pool mode.
    initial_pool_size : int, optional
        If ``use_pool_allocator=True``, sets initial pool size.
        If ``None``, us
es 1/2 of total GPU memory.
    enable_logging : bool, optional
        Enable logging (default ``False``).
        Enabling this option will introduce performance overhead.
    """
    rmm.finalize()
    rmm_config.use_managed_memory = use_managed_memory
    if use_pool_allocator:
        rmm_config.use_pool_allocator = use_pool_allocator
        if initial_pool_size is None:
            initial_pool_size = 0  # 0 means 1/2 GPU memory
        elif initial_pool_size == 0:
            initial_pool_size = 1  # Since "0" is semantic value, use 1 byte
        if not isinstance(initial_pool_size, int):
            raise TypeError("initial_pool_size must be an integer")
        rmm_config.initial_pool_size = initial_pool_size
    rmm_config.enable_logging = enable_logging
    rmm.initialize()


@initfunc
def set_allocator(allocator="default", pool=False, initial_pool_size=None):
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
    """
    use_managed_memory = True if allocator == "managed" else False
    _set_rmm_config(use_managed_memory, pool, initial_pool_size)


IS_NEP18_ACTIVE = _is_nep18_active()
_have_cupy = _have_cupy()
