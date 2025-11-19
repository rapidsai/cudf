# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import decimal
import functools
import os
import traceback
import warnings
from typing import Any

import numpy as np
import pandas as pd

import cudf

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


def is_na_like(obj: Any) -> bool:
    """
    Check if `obj` is a cudf NA value,
    i.e., None, cudf.NA or cudf.NaT
    """
    return obj is None or obj is pd.NA or obj is pd.NaT


def _is_null_host_scalar(slr: Any) -> bool:
    # slr is NA like or NaT like
    return (
        is_na_like(slr)
        or (isinstance(slr, (np.datetime64, np.timedelta64)) and np.isnat(slr))
        or slr is pd.NaT
    )


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


def _is_same_name(left_name: Any, right_name: Any) -> bool:
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
