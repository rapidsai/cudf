# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import decimal
import functools
import os
import traceback
from typing import FrozenSet, Set, Union

import cupy as cp
import numpy as np
import pandas as pd

import rmm

import cudf
from cudf.core import column
from cudf.core.buffer import Buffer
from cudf.utils.dtypes import to_cudf_compatible_scalar

# dummy commit


# The size of the mask in bytes
mask_dtype = cudf.dtype(np.int32)
mask_bitsize = mask_dtype.itemsize * 8


_EQUALITY_OPS = {
    "__eq__",
    "__ne__",
    "__lt__",
    "__gt__",
    "__le__",
    "__ge__",
}


# The test root is set by pytest to support situations where tests are run from
# a source tree on a built version of cudf.
NO_EXTERNAL_ONLY_APIS = os.getenv("NO_EXTERNAL_ONLY_APIS")

_cudf_root = os.path.dirname(cudf.__file__)
# If the environment variable for the test root is not set, we default to
# using the path relative to the cudf root directory.
_tests_root = os.getenv("_CUDF_TEST_ROOT") or os.path.join(_cudf_root, "tests")


def _external_only_api(func, alternative=""):
    """Decorator to indicate that a function should not be used internally.

    cudf contains many APIs that exist for pandas compatibility but are
    intrinsically inefficient. For some of these cudf has internal
    equivalents that are much faster. Usage of the slow public APIs inside
    our implementation can lead to unnecessary performance bottlenecks.
    Applying this decorator to such functions and setting the environment
    variable NO_EXTERNAL_ONLY_APIS will cause such functions to raise
    exceptions if they are called from anywhere inside cudf, making it easy
    to identify and excise such usage.

    The `alternative` should be a complete phrase or sentence since it will
    be used verbatim in error messages.
    """

    # If the first arg is a string then an alternative function to use in
    # place of this API was provided, so we pass that to a subsequent call.
    # It would be cleaner to implement this pattern by using a class
    # decorator with a factory method, but there is no way to generically
    # wrap docstrings on a class (we would need the docstring to be on the
    # class itself, not instances, because that's what `help` looks at) and
    # there is also no way to make mypy happy with that approach.
    if isinstance(func, str):
        return lambda actual_func: _external_only_api(actual_func, func)

    if not NO_EXTERNAL_ONLY_APIS:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check the immediately preceding frame to see if it's in cudf.
        frame, lineno = next(traceback.walk_stack(None))
        fn = frame.f_code.co_filename
        if _cudf_root in fn and _tests_root not in fn:
            raise RuntimeError(
                f"External-only API called in {fn} at line {lineno}. "
                f"{alternative}"
            )
        return func(*args, **kwargs)

    return wrapper


# TODO: We should evaluate whether calls to this could be more easily replaced
# with column.full, which appears to be significantly faster in simple cases.
def scalar_broadcast_to(scalar, size, dtype=None):

    if isinstance(size, (tuple, list)):
        size = size[0]

    if cudf._lib.scalar._is_null_host_scalar(scalar):
        if dtype is None:
            dtype = "object"
        return column.column_empty(size, dtype=dtype, masked=True)

    if isinstance(scalar, pd.Categorical):
        if dtype is None:
            return _categorical_scalar_broadcast_to(scalar, size)
        else:
            return scalar_broadcast_to(scalar.categories[0], size).astype(
                dtype
            )

    if isinstance(scalar, decimal.Decimal):
        if dtype is None:
            dtype = cudf.Decimal128Dtype._from_decimal(scalar)

        out_col = column.column_empty(size, dtype=dtype)
        if out_col.size != 0:
            out_col[:] = scalar
        return out_col

    scalar = to_cudf_compatible_scalar(scalar, dtype=dtype)
    dtype = scalar.dtype

    return cudf.core.column.full(size=size, fill_value=scalar, dtype=dtype)


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


class GetAttrGetItemMixin:
    """This mixin changes `__getattr__` to attempt a `__getitem__` call.

    Classes that include this mixin gain enhanced functionality for the
    behavior of attribute access like `obj.foo`: if `foo` is not an attribute
    of `obj`, obj['foo'] will be attempted, and the result returned.  To make
    this behavior safe, classes that include this mixin must define a class
    attribute `_PROTECTED_KEYS` that defines the attributes that are accessed
    within `__getitem__`. For example, if `__getitem__` is defined as
    `return self._data[key]`, we must define `_PROTECTED_KEYS={'_data'}`.
    """

    # Tracking of protected keys by each subclass is necessary to make the
    # `__getattr__`->`__getitem__` call safe. See
    # https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html  # noqa: E501
    # for an explanation. In brief, defining the `_PROTECTED_KEYS` allows this
    # class to avoid calling `__getitem__` inside `__getattr__` when
    # `__getitem__` will internally again call `__getattr__`, resulting in an
    # infinite recursion.
    # This problem only arises when the copy protocol is invoked (e.g. by
    # `copy.copy` or `pickle.dumps`), and could also be avoided by redefining
    # methods involved with the copy protocol such as `__reduce__` or
    # `__setstate__`, but this class may be used in complex multiple
    # inheritance hierarchies that might also override serialization.  The
    # solution here is a minimally invasive change that avoids such conflicts.
    _PROTECTED_KEYS: Union[FrozenSet[str], Set[str]] = frozenset()

    def __getattr__(self, key):
        if key in self._PROTECTED_KEYS:
            raise AttributeError
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {key}"
            )


class NotIterable:
    def __iter__(self):
        raise TypeError(
            f"{self.__class__.__name__} object is not iterable. "
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


def _isnat(val):
    """Wraps np.isnat to return False instead of error on invalid inputs."""
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


def _categorical_scalar_broadcast_to(cat_scalar, size):
    if isinstance(cat_scalar, (cudf.Series, pd.Series)):
        cats = cat_scalar.cat.categories
        code = cat_scalar.cat.codes[0]
        ordered = cat_scalar.cat.ordered
    else:
        # handles pd.Categorical, cudf.categorical.CategoricalColumn
        cats = cat_scalar.categories
        code = cat_scalar.codes[0]
        ordered = cat_scalar.ordered

    cats = column.as_column(cats)
    codes = scalar_broadcast_to(code, size)

    return column.build_categorical_column(
        categories=cats,
        codes=codes,
        mask=codes.base_mask,
        size=codes.size,
        offset=codes.offset,
        ordered=ordered,
    )


def _create_pandas_series(
    data=None, index=None, dtype=None, name=None, copy=False, fastpath=False
):
    """
    Wrapper to create a Pandas Series. If the length of data is 0 and
    dtype is not passed, this wrapper defaults the dtype to `float64`.

    Parameters
    ----------
    data : array-like, Iterable, dict, or scalar value
        Contains data stored in Series. If data is a dict, argument
        order is maintained.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as data.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, …, n) if not provided.
        If data is dict-like and index is None, then the keys
        in the data are used as the index. If the index is not None,
        the resulting Series is reindexed with the index values.
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Series. If not specified, this
        will be inferred from data. See the user guide for more usages.
    name : str, optional
        The name to give to the Series.
    copy : bool, default False
        Copy input data.

    Returns
    -------
    pd.Series
    """
    if (data is None or len(data) == 0) and dtype is None:
        dtype = "float64"
    return pd.Series(
        data=data,
        index=index,
        dtype=dtype,
        name=name,
        copy=copy,
        fastpath=fastpath,
    )


def _maybe_indices_to_slice(indices: cp.ndarray) -> Union[slice, cp.ndarray]:
    """Makes best effort to convert an array of indices into a python slice.
    If the conversion is not possible, return input. `indices` are expected
    to be valid.
    """
    # TODO: improve efficiency by avoiding sync.
    if len(indices) == 1:
        x = indices[0].item()
        return slice(x, x + 1)
    if len(indices) == 2:
        x1, x2 = indices[0].item(), indices[1].item()
        return slice(x1, x2 + 1, x2 - x1)
    start, step = indices[0].item(), (indices[1] - indices[0]).item()
    stop = start + step * len(indices)
    if (indices == cp.arange(start, stop, step)).all():
        return slice(start, stop, step)
    return indices
