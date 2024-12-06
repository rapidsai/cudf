# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

import decimal
import functools
import os
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd

import pylibcudf as plc
import rmm

import cudf
import cudf.api.types
from cudf.core import column
from cudf.core.buffer import as_buffer

# The size of the mask in bytes
mask_dtype = cudf.api.types.dtype(np.int32)
mask_bitsize = mask_dtype.itemsize * 8

# Mapping from ufuncs to the corresponding binary operators.
_ufunc_binary_operations = {
    # Arithmetic binary operations.
    "add": "add",
    "subtract": "sub",
    "multiply": "mul",
    "matmul": "matmul",
    "divide": "truediv",
    "true_divide": "truediv",
    "floor_divide": "floordiv",
    "power": "pow",
    "float_power": "pow",
    "remainder": "mod",
    "mod": "mod",
    "fmod": "mod",
    # Bitwise binary operations.
    "bitwise_and": "and",
    "bitwise_or": "or",
    "bitwise_xor": "xor",
    # Comparison binary operators
    "greater": "gt",
    "greater_equal": "ge",
    "less": "lt",
    "less_equal": "le",
    "not_equal": "ne",
    "equal": "eq",
}

# These operators need to be mapped to their inverses when performing a
# reflected ufunc operation because no reflected version of the operators
# themselves exist. When these operators are invoked directly (not via
# __array_ufunc__) Python takes care of calling the inverse operation.
_ops_without_reflection = {
    "gt": "lt",
    "ge": "le",
    "lt": "gt",
    "le": "ge",
    # ne and eq are symmetric, so they are their own inverse op
    "ne": "ne",
    "eq": "eq",
}


# This is the implementation of __array_ufunc__ used for Frame and Column.
# For more detail on this function and how it should work, see
# https://numpy.org/doc/stable/reference/ufuncs.html
def _array_ufunc(obj, ufunc, method, inputs, kwargs):
    # We don't currently support reduction, accumulation, etc. We also
    # don't support any special kwargs or higher arity ufuncs than binary.
    if method != "__call__" or kwargs or ufunc.nin > 2:
        return NotImplemented

    fname = ufunc.__name__
    if fname in _ufunc_binary_operations:
        reflect = obj is not inputs[0]
        other = inputs[0] if reflect else inputs[1]

        op = _ufunc_binary_operations[fname]
        if reflect and op in _ops_without_reflection:
            op = _ops_without_reflection[op]
            reflect = False
        op = f"__{'r' if reflect else ''}{op}__"

        # float_power returns float irrespective of the input type.
        # TODO: Do not get the attribute directly, get from the operator module
        # so that we can still exploit reflection.
        if fname == "float_power":
            return getattr(obj, op)(other).astype(float)
        return getattr(obj, op)(other)

    # Special handling for various unary operations.
    if fname == "negative":
        return obj * -1
    if fname == "positive":
        return obj.copy(deep=True)
    if fname == "invert":
        return ~obj
    if fname == "absolute":
        # TODO: Make sure all obj (mainly Column) implement abs.
        return abs(obj)
    if fname == "fabs":
        return abs(obj).astype(np.float64)

    # None is a sentinel used by subclasses to trigger cupy dispatch.
    return None


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
        pre_frame = traceback.extract_stack(limit=2)[0]
        fn = pre_frame.filename
        lineno = pre_frame.lineno
        if _cudf_root in fn and _tests_root not in fn:
            raise RuntimeError(
                f"External-only API called in {fn} at line {lineno}. "
                f"{alternative}"
            )
        return func(*args, **kwargs)

    return wrapper


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


def clear_cache():
    """Clear all internal caches"""
    cudf.Scalar._clear_instance_cache()


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
    # https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
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
    _PROTECTED_KEYS: frozenset[str] | set[str] = frozenset()

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
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            f"{self.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.values_host` "
            f"if you wish to iterate over the values."
        )


def pa_mask_buffer_to_mask(mask_buf, size):
    """
    Convert PyArrow mask buffer to cuDF mask buffer
    """
    mask_size = plc.null_mask.bitmask_allocation_size_bytes(size)
    if mask_buf.size < mask_size:
        dbuf = rmm.DeviceBuffer(size=mask_size)
        dbuf.copy_from_host(np.asarray(mask_buf).view("u1"))
        return as_buffer(dbuf)
    return as_buffer(mask_buf)


def _isnat(val):
    """Wraps np.isnat to return False instead of error on invalid inputs."""
    if val is pd.NaT:
        return True
    elif not isinstance(val, (np.datetime64, np.timedelta64, str)):
        return False
    else:
        try:
            return val in {"NaT", "NAT"} or np.isnat(val)
        except TypeError:
            return False


def search_range(x: int, ri: range, *, side: str) -> int:
    """

    Find insertion point in a range to maintain sorted order

    Parameters
    ----------
    x
        Integer to insert
    ri
        Range to insert into
    side
        Tie-breaking decision for the case that `x` is a member of the
        range. If `"left"` then the insertion point is before the
        entry, otherwise it is after.

    Returns
    -------
    int
        The insertion point

    See Also
    --------
    numpy.searchsorted

    Notes
    -----
    Let ``p`` be the return value, then if ``side="left"`` the
    following invariants are maintained::

        all(x < n for n in ri[:p])
        all(x >= n for n in ri[p:])

    Conversely, if ``side="right"`` then we have::

        all(x <= n for n in ri[:p])
        all(x > n for n in ri[p:])

    Examples
    --------
    For series: 1 4 7
    >>> search_range(4, range(1, 10, 3), side="left")
    1
    >>> search_range(4, range(1, 10, 3), side="right")
    2
    """
    assert side in {"left", "right"}
    if flip := (ri.step < 0):
        ri = ri[::-1]
        shift = int(side == "right")
    else:
        shift = int(side == "left")

    offset = (x - ri.start - shift) // ri.step + 1
    if flip:
        offset = len(ri) - offset
    return max(min(len(ri), offset), 0)


def is_na_like(obj):
    """
    Check if `obj` is a cudf NA value,
    i.e., None, cudf.NA or cudf.NaT
    """
    return obj is None or obj is cudf.NA or obj is cudf.NaT


def _warn_no_dask_cudf(fn):
    @functools.wraps(fn)
    def wrapper(self):
        # try import
        try:
            # Import dask_cudf (if available) in case
            # this is being called within Dask Dataframe
            import dask_cudf  # noqa: F401

        except ImportError:
            warnings.warn(
                f"Using dask to tokenize a {type(self)} object, "
                "but `dask_cudf` is not installed. Please install "
                "`dask_cudf` for proper dispatching."
            )
        return fn(self)

    return wrapper


def _is_same_name(left_name, right_name):
    # Internal utility to compare if two names are same.
    with warnings.catch_warnings():
        # numpy throws warnings while comparing
        # NaT values with non-NaT values.
        warnings.simplefilter("ignore")
        try:
            same = (left_name is right_name) or (left_name == right_name)
            if not same:
                if isinstance(left_name, decimal.Decimal) and isinstance(
                    right_name, decimal.Decimal
                ):
                    return left_name.is_nan() and right_name.is_nan()
                if isinstance(left_name, float) and isinstance(
                    right_name, float
                ):
                    return np.isnan(left_name) and np.isnan(right_name)
                if isinstance(left_name, np.datetime64) and isinstance(
                    right_name, np.datetime64
                ):
                    return np.isnan(left_name) and np.isnan(right_name)
            return same
        except TypeError:
            return False


def _all_bools_with_nulls(lhs, rhs, bool_fill_value):
    # Internal utility to construct a boolean column
    # by combining nulls from `lhs` & `rhs`.
    if lhs.has_nulls() and rhs.has_nulls():
        result_mask = lhs._get_mask_as_column() & rhs._get_mask_as_column()
    elif lhs.has_nulls():
        result_mask = lhs._get_mask_as_column()
    elif rhs.has_nulls():
        result_mask = rhs._get_mask_as_column()
    else:
        result_mask = None

    result_col = column.as_column(
        bool_fill_value, dtype=cudf.dtype(np.bool_), length=len(lhs)
    )
    if result_mask is not None:
        result_col = result_col.set_mask(result_mask.as_mask())
    return result_col


def _datetime_timedelta_find_and_replace(
    original_column: "cudf.core.column.DatetimeColumn"
    | "cudf.core.column.TimeDeltaColumn",
    to_replace: Any,
    replacement: Any,
    all_nan: bool = False,
) -> "cudf.core.column.DatetimeColumn" | "cudf.core.column.TimeDeltaColumn":
    """
    This is an internal utility to find and replace values in a datetime or
    timedelta column. It is used by the `find_and_replace` method of
    `DatetimeColumn` and `TimeDeltaColumn`. Centralizing the code in a single
    as opposed to duplicating it in both classes.
    """
    original_col_class = type(original_column)
    if not isinstance(to_replace, original_col_class):
        to_replace = cudf.core.column.as_column(to_replace)
        if to_replace.can_cast_safely(original_column.dtype):
            to_replace = to_replace.astype(original_column.dtype)
    if not isinstance(replacement, original_col_class):
        replacement = cudf.core.column.as_column(replacement)
        if replacement.can_cast_safely(original_column.dtype):
            replacement = replacement.astype(original_column.dtype)
    if isinstance(to_replace, original_col_class):
        to_replace = to_replace.as_numerical_column(dtype=np.dtype("int64"))
    if isinstance(replacement, original_col_class):
        replacement = replacement.as_numerical_column(dtype=np.dtype("int64"))
    try:
        result_col = (
            original_column.as_numerical_column(dtype=np.dtype("int64"))
            .find_and_replace(to_replace, replacement, all_nan)
            .astype(original_column.dtype)
        )
    except TypeError:
        result_col = original_column.copy(deep=True)
    return result_col  # type: ignore
