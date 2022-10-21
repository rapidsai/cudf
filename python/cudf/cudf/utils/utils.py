# Copyright (c) 2020-2022, NVIDIA CORPORATION.

import functools
import hashlib
import os
import traceback
import warnings
from functools import partial
from typing import FrozenSet, Set, Union

import numpy as np
from nvtx import annotate

import rmm

import cudf
import cudf.api.types
from cudf.core import column
from cudf.core.buffer import as_device_buffer_like

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

_NVTX_COLORS = ["green", "blue", "purple", "rapids"]

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
    warnings.warn(
        "The cudf.set_allocator function is deprecated and will be removed in "
        "a future release. Please use rmm.reinitialize "
        "(https://docs.rapids.ai/api/rmm/stable/api.html#rmm.reinitialize) "
        'instead. Note that `cudf.set_allocator(allocator="managed")` is '
        "equivalent to `rmm.reinitialize(managed_memory=True)`.",
        FutureWarning,
    )

    use_managed_memory = allocator == "managed"

    rmm.reinitialize(
        pool_allocator=pool,
        managed_memory=use_managed_memory,
        initial_pool_size=initial_pool_size,
        logging=enable_logging,
    )


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
        return as_device_buffer_like(dbuf)
    return as_device_buffer_like(mask_buf)


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
    ----------
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
    -------
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


def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]


def _cudf_nvtx_annotate(func, domain="cudf_python"):
    """Decorator for applying nvtx annotations to methods in cudf."""
    return annotate(
        message=func.__qualname__,
        color=_get_color_for_nvtx(func.__qualname__),
        domain=domain,
    )(func)


_dask_cudf_nvtx_annotate = partial(
    _cudf_nvtx_annotate, domain="dask_cudf_python"
)
