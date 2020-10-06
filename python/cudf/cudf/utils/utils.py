# Copyright (c) 2020, NVIDIA CORPORATION.
import functools
from collections import OrderedDict
from math import floor, isinf, isnan

import numpy as np
import pandas as pd
from numba import njit

import rmm

import cudf
from cudf.core import column
from cudf.core.buffer import Buffer
from cudf.utils.dtypes import to_cudf_compatible_scalar

mask_dtype = np.dtype(np.int32)
mask_bitsize = mask_dtype.itemsize * 8


@njit
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


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


def scalar_broadcast_to(scalar, size, dtype=None):

    if isinstance(size, (tuple, list)):
        size = size[0]

    if scalar is None or (
        isinstance(scalar, (np.datetime64, np.timedelta64))
        and np.isnat(scalar)
    ):
        if dtype is None:
            dtype = "object"
        return column.column_empty(size, dtype=dtype, masked=True)

    if isinstance(scalar, pd.Categorical):
        return scalar_broadcast_to(scalar.categories[0], size).astype(dtype)

    scalar = to_cudf_compatible_scalar(scalar, dtype=dtype)
    dtype = scalar.dtype

    if np.dtype(dtype).kind in ("O", "U"):
        gather_map = column.full(size, 0, dtype="int32")
        scalar_str_col = column.as_column([scalar], dtype="str")
        return scalar_str_col[gather_map]
    else:
        out_col = column.column_empty(size, dtype=dtype)
        if out_col.size != 0:
            out_col.data_array_view[:] = scalar
        return out_col


def normalize_index(index, size, doraise=True):
    """Normalize negative index
    """
    if index < 0:
        index = size + index
    if doraise and not (0 <= index < size):
        raise IndexError("out-of-bound")
    return min(index, size)


list_types_tuple = (list, np.array)


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

    if isinstance(right, (cudf.Series, cudf.Index, pd.Series, pd.Index)):
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

    empty_col = column.column_empty(size, dtype, True)
    return cudf.Series(empty_col)


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


class ColumnValuesMappingMixin:
    """
    Coerce provided values for the mapping to Columns.
    """

    def __setitem__(self, key, value):

        value = column.as_column(value)
        super().__setitem__(key, value)


class EqualLengthValuesMappingMixin:
    """
    Require all values in the mapping to have the same length.
    """

    def __setitem__(self, key, value):
        if len(self) > 0:
            first = next(iter(self.values()))
            if len(value) != len(first):
                raise ValueError("All values must be of equal length")
        super().__setitem__(key, value)


class OrderedColumnDict(
    ColumnValuesMappingMixin, EqualLengthValuesMappingMixin, OrderedDict
):
    pass


class NestedMappingMixin:
    """
    Make missing values of a mapping empty instances
    of the same type as the mapping.
    """

    def __getitem__(self, key):
        if isinstance(key, tuple):
            d = self
            for k in key[:-1]:
                d = d[k]
            return d.__getitem__(key[-1])
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            d = self
            for k in key[:-1]:
                d = d.setdefault(k, self.__class__())
            d.__setitem__(key[-1], value)
        else:
            super().__setitem__(key, value)


class NestedOrderedDict(NestedMappingMixin, OrderedDict):
    pass


def to_flat_dict(d):
    """
    Convert the given nested dictionary to a flat dictionary
    with tuple keys.
    """

    def _inner(d, parents=None):
        if parents is None:
            parents = []
        for k, v in d.items():
            if not isinstance(v, d.__class__):
                if parents:
                    k = tuple(parents + [k])
                yield (k, v)
            else:
                yield from _inner(d=v, parents=parents + [k])

    return {k: v for k, v in _inner(d)}


def to_nested_dict(d):
    """
    Convert the given dictionary with tuple keys to a NestedOrderedDict.
    """
    return NestedOrderedDict(d)


def time_col_replace_nulls(input_col):

    null = column.column_empty_like(input_col, masked=True, newsize=1)
    out_col = cudf._lib.replace.replace(
        input_col,
        column.as_column(
            Buffer(
                np.array(
                    [input_col.default_na_value()], dtype=input_col.dtype
                ).view("|u1")
            ),
            dtype=input_col.dtype,
        ),
        null,
    )
    return out_col


def raise_iteration_error(obj):
    raise TypeError(
        f"{obj.__class__.__name__} object is not iterable. "
        f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
        f"if you wish to iterate over the values."
    )


def pa_mask_buffer_to_mask(mask_buf, size):
    """
    Convert PyArrow mask buffer to cuDF mask buffer
    """
    mask_size = cudf._lib.null_mask.bitmask_allocation_size_bytes(size)
    if mask_buf.size < mask_size:
        dbuf = rmm.DeviceBuffer(size=mask_size)
        dbuf.copy_from_host(np.asarray(mask_buf).view("u1"))
        return Buffer(dbuf)
    return Buffer(mask_buf)
