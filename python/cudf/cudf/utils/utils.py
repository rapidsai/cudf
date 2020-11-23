# Copyright (c) 2020, NVIDIA CORPORATION.
import functools
from collections import OrderedDict
from collections.abc import Sequence
from math import floor, isinf, isnan

import numpy as np
import cupy as cp
import pandas as pd
from numba import njit

import rmm

import cudf
from cudf.core import column
from cudf.core.buffer import Buffer
from cudf.utils.dtypes import to_cudf_compatible_scalar

mask_dtype = np.dtype(np.int32)
mask_bitsize = mask_dtype.itemsize * 8

_EQUALITY_OPS = {
    "eq",
    "ne",
    "lt",
    "gt",
    "le",
    "ge",
    "__eq__",
    "__ne__",
    "__lt__",
    "__gt__",
    "__le__",
    "__ge__",
}


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


def isnat(val):
    if not isinstance(val, (np.datetime64, np.timedelta64, str)):
        return False
    else:
        return val in {"NaT", "NAT"} or np.isnat(val)


def _fillna_natwise(col):
    # If the value we are filling is np.datetime64("NAT")
    # we set the same mask as current column.
    # However where there are "<NA>" in the
    # columns, their corresponding locations
    nat = cudf._lib.scalar._create_proxy_nat_scalar(col.dtype)
    result = cudf._lib.replace.replace_nulls(col, nat)
    return column.build_column(
        data=result.base_data,
        dtype=result.dtype,
        mask=col.base_mask,
        size=result.size,
        offset=result.offset,
        children=result.base_children,
    )


def search_range(start, stop, x, step=1, side="left"):
    """Find the position to insert a value in a range, so that the resulting
    sequence remains sorted.

    When ``side`` is set to 'left', the insertion point ``i`` will hold the
    following invariant:
    `all(x < n for x in range_left) and all(x >= n for x in range_right)`
    where ``range_left`` and ``range_right`` refers to the range to the left
    and right of position ``i``, respectively.

    When ``side`` is set to 'right', ``i`` will hold the following invariant:
    `all(x <= n for x in range_left) and all(x > n for x in range_right)`

    Parameters
    --------
    start : int
        Start value of the series
    stop : int
        Stop value of the range
    x : int
        The value to insert
    step : int, default 1
        Step value of the series, assumed positive
    side : {'left', 'right'}, default 'left'
        See description for usage.

    Returns
    --------
    int
        Insertion position of n.

    Examples
    --------
    For series: 1 4 7
    >>> search_range(start=1, stop=10, x=4, step=3, side="left")
    1
    >>> search_range(start=1, stop=10, x=4, step=3, side="right")
    2
    """
    z = 1 if side == "left" else 0
    i = (x - start - z) // step + 1

    length = (stop - start) // step
    return max(min(length, i), 0)


# Utils for using appropriate dispatch for array functions
def get_appropriate_dispatched_func(
    cudf_submodule, cudf_ser_submodule, cupy_submodule, func, args, kwargs
):
    fname = func.__name__

    if hasattr(cudf_submodule, fname):
        cudf_func = getattr(cudf_submodule, fname)
        return cudf_func(*args, **kwargs)

    elif hasattr(cudf_ser_submodule, fname):
        cudf_ser_func = getattr(cudf_ser_submodule, fname)
        return cudf_ser_func(*args, **kwargs)

    elif hasattr(cupy_submodule, fname):
        cupy_func = getattr(cupy_submodule, fname)
        # Handle case if cupy impliments it as a numpy function
        # Unsure if needed
        if cupy_func is func:
            return NotImplemented

        cupy_compatible_args = get_cupy_compatible_args(args)
        cupy_output = cupy_func(*cupy_compatible_args, **kwargs)
        return cast_to_appropriate_cudf_type(cupy_output)
    else:
        return NotImplemented


def cast_to_appropriate_cudf_type(val):
    # TODO Handle scalar
    if val.ndim == 0:
        return cudf.Scalar(val).value
    # 1D array
    elif (val.ndim == 1) or (val.ndim == 2 and val.shape[1] == 1):
        return cudf.Series(val)
    else:
        return NotImplemented


def get_cupy_compatible_args(args):
    casted_ls = []
    for arg in args:
        if isinstance(arg, cp.ndarray):
            casted_ls.append(arg)
        elif isinstance(arg, cudf.Series):
            casted_ls.append(arg.values)
        elif isinstance(arg, Sequence):
            # handle list of inputs for functions like
            # np.concatenate
            casted_arg = get_cupy_compatible_args(arg)
            casted_ls.append(casted_arg)
        else:
            casted_ls.append(arg)
    return casted_ls


def get_relevant_submodule(func, module):
    # point to the correct submodule
    for submodule in func.__module__.split(".")[1:]:
        if hasattr(module, submodule):
            module = getattr(module, submodule)
        else:
            return None
    return module
