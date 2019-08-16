import numbers
from collections import namedtuple
from math import ceil, isinf, isnan

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import njit

from librmm_cffi import librmm as rmm

from cudf.utils.dtypes import is_categorical_dtype

mask_dtype = np.dtype(np.int8)
mask_bitsize = mask_dtype.itemsize * 8
mask_byte_padding = 64


def calc_chunk_size(size, chunksize):
    return mask_byte_padding * ceil(
        ((size + chunksize - 1) // chunksize) / mask_byte_padding
    )


_TypeMinMax = namedtuple("_TypeMinMax", "min,max")


def get_numeric_type_info(dtype):
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind in "f":
        return _TypeMinMax(dtype.type("-inf"), dtype.type("+inf"))
    else:
        raise TypeError(dtype)


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

    scalar = to_cudf_compatible_scalar(scalar, dtype=dtype)

    if not isinstance(shape, tuple):
        shape = (shape,)

    if np.dtype(dtype) == np.dtype("object"):
        import nvstrings
        from cudf.dataframe.string import StringColumn
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
    from cudf.dataframe.buffer import Buffer
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


def cudf_dtype_from_pydata_dtype(dtype):
    """ Given a numpy or pandas dtype, converts it into the equivalent cuDF
        Python dtype.
    """
    try:
        # pd 0.24.X
        from pandas.core.dtypes.common import infer_dtype_from_object
    except ImportError:
        # pd 0.23.X
        from pandas.core.dtypes.common import (
            _get_dtype_from_object as infer_dtype_from_object,
        )

    if is_categorical_dtype(dtype):
        pass
    elif np.issubdtype(dtype, np.datetime64):
        dtype = np.datetime64

    return infer_dtype_from_object(dtype)


def is_scalar(val):
    return (
        val is None
        or isinstance(val, str)
        or isinstance(val, numbers.Number)
        or np.isscalar(val)
        or isinstance(val, pd.Timestamp)
        or isinstance(val, pd.Categorical)
    )


def to_cudf_compatible_scalar(val, dtype=None):
    """
    Converts the value `val` to a numpy/Pandas scalar,
    optionally casting to `dtype`.

    If `val` is None, returns None.
    """
    if val is None:
        return val

    if not is_scalar(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} "
            " to cudf scalar"
        )

    val = pd.api.types.pandas_dtype(type(val)).type(val)

    if dtype is not None:
        val = val.astype(dtype)

    return val


def is_list_like(obj):
    """
    This function checks if the given `obj`
    is a list-like (list, tuple, Series...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is like-like or not.
    """
    from collections.abc import Sequence

    if isinstance(obj, (Sequence,)) and not isinstance(obj, (str, bytes)):
        return True
    else:
        return False


def min_scalar_type(a, min_size=8):
    return min_signed_type(a, min_size=min_size)


def min_signed_type(x, min_size=8):
    """
    Return the smallest *signed* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in np.sctypes["int"]:
        if (np.dtype(int_dtype).itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return int_dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


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
