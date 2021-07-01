# Copyright (c) 2020-2021, NVIDIA CORPORATION.

import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from decimal import Decimal

import cupy
import numpy as np
import pandas as pd
import pytest
from pandas import testing as tm

import cudf
from cudf._lib.null_mask import bitmask_allocation_size_bytes
from cudf.core.column.datetime import _numpy_to_pandas_conversion
from cudf.utils import dtypes as dtypeutils

supported_numpy_dtypes = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "datetime64[ms]",
    "datetime64[us]",
]

SIGNED_INTEGER_TYPES = sorted(list(dtypeutils.SIGNED_INTEGER_TYPES))
UNSIGNED_TYPES = sorted(list(dtypeutils.UNSIGNED_TYPES))
INTEGER_TYPES = sorted(list(dtypeutils.INTEGER_TYPES))
FLOAT_TYPES = sorted(list(dtypeutils.FLOAT_TYPES))
SIGNED_TYPES = sorted(list(dtypeutils.SIGNED_TYPES))
NUMERIC_TYPES = sorted(list(dtypeutils.NUMERIC_TYPES))
DATETIME_TYPES = sorted(list(dtypeutils.DATETIME_TYPES))
TIMEDELTA_TYPES = sorted(list(dtypeutils.TIMEDELTA_TYPES))
OTHER_TYPES = sorted(list(dtypeutils.OTHER_TYPES))
ALL_TYPES = sorted(list(dtypeutils.ALL_TYPES))


def random_bitmask(size):
    """
    Parameters
    ----------
    size : int
        number of bits
    """
    sz = bitmask_allocation_size_bytes(size)
    data = np.random.randint(0, 255, dtype="u1", size=sz)
    return data.view("i1")


def expand_bits_to_bytes(arr):
    def fix_binary(bstr):
        bstr = bstr[2:]
        diff = 8 - len(bstr)
        return ("0" * diff + bstr)[::-1]

    ba = bytearray(arr.data)
    return list(map(int, "".join(map(fix_binary, map(bin, ba)))))


def count_zero(arr):
    arr = np.asarray(arr)
    return np.count_nonzero(arr == 0)


def assert_eq(left, right, **kwargs):
    """ Assert that two cudf-like things are equivalent

    This equality test works for pandas/cudf dataframes/series/indexes/scalars
    in the same way, and so makes it easier to perform parametrized testing
    without switching between assert_frame_equal/assert_series_equal/...
    functions.
    """
    # dtypes that we support but Pandas doesn't will convert to
    # `object`. Check equality before that happens:
    if kwargs.get("check_dtype", True):
        if hasattr(left, "dtype") and hasattr(right, "dtype"):
            if isinstance(
                left.dtype, cudf.core.dtypes._BaseDtype
            ) and not isinstance(
                left.dtype, cudf.CategoricalDtype
            ):  # leave categorical comparison to Pandas
                assert_eq(left.dtype, right.dtype)

    if hasattr(left, "to_pandas"):
        left = left.to_pandas()
    if hasattr(right, "to_pandas"):
        right = right.to_pandas()
    if isinstance(left, cupy.ndarray):
        left = cupy.asnumpy(left)
    if isinstance(right, cupy.ndarray):
        right = cupy.asnumpy(right)

    if isinstance(left, pd.DataFrame):
        tm.assert_frame_equal(left, right, **kwargs)
    elif isinstance(left, pd.Series):
        tm.assert_series_equal(left, right, **kwargs)
    elif isinstance(left, pd.Index):
        tm.assert_index_equal(left, right, **kwargs)
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if np.issubdtype(left.dtype, np.floating) and np.issubdtype(
            right.dtype, np.floating
        ):
            assert np.allclose(left, right, equal_nan=True)
        else:
            assert np.array_equal(left, right)
    else:
        # Use the overloaded __eq__ of the operands
        if left == right:
            return True
        elif any([np.issubdtype(type(x), np.floating) for x in (left, right)]):
            np.testing.assert_almost_equal(left, right)
        else:
            np.testing.assert_equal(left, right)
    return True


def assert_neq(left, right, **kwargs):
    __tracebackhide__ = True
    try:
        assert_eq(left, right, **kwargs)
    except AssertionError:
        pass
    else:
        raise AssertionError


def assert_exceptions_equal(
    lfunc,
    rfunc,
    lfunc_args_and_kwargs=None,
    rfunc_args_and_kwargs=None,
    check_exception_type=True,
    compare_error_message=True,
    expected_error_message=None,
):
    """Compares if two functions ``lfunc`` and ``rfunc`` raise
    same exception or not.

    Parameters
    ----------
    lfunc : callable
        A callable function to obtain the Exception.
    rfunc : callable
        A callable function to compare the Exception
        obtained by calling ``rfunc``.
    lfunc_args_and_kwargs : tuple, default None
        Tuple containing positional arguments at first position,
        and key-word arguments at second position that need to be passed into
        ``lfunc``. If the tuple is of length 1, it must either contain
        positional arguments(as a Sequence) or key-word arguments(as a Mapping
        dict).
    rfunc_args_and_kwargs : tuple, default None
        Tuple containing positional arguments at first position,
        and key-word arguments at second position that need to be passed into
        ``rfunc``. If the tuple is of length 1, it must either contain
        positional arguments(as a Sequence) or key-word arguments(as a Mapping
        dict).
    check_exception_type : boolean, default True
        Whether to compare the exception types raised by ``lfunc``
        with ``rfunc`` exception type or not. If False, ``rfunc``
        is simply evaluated against `Exception` type.
    compare_error_message : boolean, default True
        Whether to compare the error messages raised
        when calling both ``lfunc`` and
        ``rfunc`` or not.
    expected_error_message : str, default None
        Expected error message to be raised by calling ``rfunc``.
        Note that ``lfunc`` error message will not be compared to
        this value.

    Returns
    -------
    None
        If exceptions raised by ``lfunc`` and
        ``rfunc`` match.

    Raises
    ------
    AssertionError
        If call to ``lfunc`` doesn't raise any Exception.
    """

    lfunc_args, lfunc_kwargs = _get_args_kwars_for_assert_exceptions(
        lfunc_args_and_kwargs
    )
    rfunc_args, rfunc_kwargs = _get_args_kwars_for_assert_exceptions(
        rfunc_args_and_kwargs
    )

    try:
        lfunc(*lfunc_args, **lfunc_kwargs)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        if not compare_error_message:
            expected_error_message = None
        elif expected_error_message is None:
            expected_error_message = re.escape(str(e))

        with pytest.raises(
            type(e) if check_exception_type else Exception,
            match=expected_error_message,
        ):
            rfunc(*rfunc_args, **rfunc_kwargs)
    else:
        raise AssertionError("Expected to fail with an Exception.")


def _get_args_kwars_for_assert_exceptions(func_args_and_kwargs):
    if func_args_and_kwargs is None:
        return [], {}
    else:
        if len(func_args_and_kwargs) == 1:
            func_args, func_kwargs = [], {}
            if isinstance(func_args_and_kwargs[0], Sequence):
                func_args = func_args_and_kwargs[0]
            elif isinstance(func_args_and_kwargs[0], Mapping):
                func_kwargs = func_args_and_kwargs[0]
            else:
                raise ValueError(
                    "length 1 func_args_and_kwargs must be "
                    "either a Sequence or a Mapping"
                )
        elif len(func_args_and_kwargs) == 2:
            if not isinstance(func_args_and_kwargs[0], Sequence):
                raise ValueError(
                    "Positional argument at 1st position of "
                    "func_args_and_kwargs should be a sequence."
                )
            if not isinstance(func_args_and_kwargs[1], Mapping):
                raise ValueError(
                    "Key-word argument at 2nd position of "
                    "func_args_and_kwargs should be a dictionary mapping."
                )

            func_args, func_kwargs = func_args_and_kwargs
        else:
            raise ValueError("func_args_and_kwargs must be of length 1 or 2")
        return func_args, func_kwargs


def gen_rand(dtype, size, **kwargs):
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        res = np.random.random(size=size).astype(dtype)
        if kwargs.get("positive_only", False):
            return res
        else:
            return res * 2 - 1
    elif dtype == np.int8 or dtype == np.int16:
        low = kwargs.get("low", -32)
        high = kwargs.get("high", 32)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "i":
        low = kwargs.get("low", -10000)
        high = kwargs.get("high", 10000)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype == np.uint8 or dtype == np.uint16:
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 32)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "u":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 128)
        return np.random.randint(low=low, high=high, size=size).astype(dtype)
    elif dtype.kind == "b":
        low = kwargs.get("low", 0)
        high = kwargs.get("high", 2)
        return np.random.randint(low=low, high=high, size=size).astype(
            np.bool_
        )
    elif dtype.kind == "M":
        low = kwargs.get("low", 0)
        time_unit, _ = np.datetime_data(dtype)
        high = kwargs.get(
            "high",
            1000000000000000000 / _numpy_to_pandas_conversion[time_unit],
        )
        return pd.to_datetime(
            np.random.randint(low=low, high=high, size=size), unit=time_unit
        )
    elif dtype.kind == "U":
        return pd.util.testing.rands_array(10, size)
    raise NotImplementedError(f"dtype.kind={dtype.kind}")


def gen_rand_series(dtype, size, **kwargs):
    values = gen_rand(dtype, size, **kwargs)
    if kwargs.get("has_nulls", False):
        return cudf.Series.from_masked_array(values, random_bitmask(size))

    return cudf.Series(values)


def _decimal_series(input, dtype):
    return cudf.Series(
        [x if x is None else Decimal(x) for x in input], dtype=dtype,
    )


@contextmanager
def does_not_raise():
    yield


def xfail_param(param, **kwargs):
    return pytest.param(param, marks=pytest.mark.xfail(**kwargs))
