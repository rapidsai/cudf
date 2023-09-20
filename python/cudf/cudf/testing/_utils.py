# Copyright (c) 2020-2023, NVIDIA CORPORATION.

import itertools
import string
import warnings
from collections import abc
from contextlib import contextmanager
from decimal import Decimal

import cupy
import numpy as np
import pandas as pd
import pytest
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudaimpl import lower as cuda_lower
from pandas import testing as tm

import cudf
from cudf._lib.null_mask import bitmask_allocation_size_bytes
from cudf.api.types import is_scalar
from cudf.core.column.timedelta import _unit_to_nanoseconds_conversion
from cudf.core.udf.strings_lowering import cast_string_view_to_udf_string
from cudf.core.udf.strings_typing import StringView, string_view, udf_string
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

SERIES_OR_INDEX_NAMES = [
    None,
    pd.NA,
    cudf.NA,
    np.nan,
    float("NaN"),
    "abc",
    1,
    pd.NaT,
    np.datetime64("nat"),
    np.timedelta64("NaT"),
    np.timedelta64(10, "D"),
    np.timedelta64(5, "D"),
    np.datetime64("1970-01-01 00:00:00.000000001"),
    np.datetime64("1970-01-01 00:00:00.000000002"),
    pd.Timestamp(1),
    pd.Timestamp(2),
    pd.Timedelta(1),
    pd.Timedelta(2),
    Decimal("NaN"),
    Decimal("1.2"),
    np.int64(1),
    np.int32(1),
    np.float32(1),
    pd.Timestamp(1),
]


def set_random_null_mask_inplace(series, null_probability=0.5, seed=None):
    """Randomly nullify elements in series with the provided probability."""
    probs = [null_probability, 1 - null_probability]
    rng = np.random.default_rng(seed=seed)
    mask = rng.choice([False, True], size=len(series), p=probs)
    series.iloc[mask] = None


# TODO: This function should be removed. Anywhere that it is being used should
# instead be generating a random boolean array (bytemask) and use the public
# APIs to set those elements to None.
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
    """Assert that two cudf-like things are equivalent

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

    if isinstance(left, (pd.DataFrame, pd.Series, pd.Index)):
        # TODO: A warning is emitted from the function
        # pandas.testing.assert_[series, frame, index]_equal for some inputs:
        # "DeprecationWarning: elementwise comparison failed; this will raise
        # an error in the future."
        # or "FutureWarning: elementwise ..."
        # This warning comes from a call from pandas to numpy. It is ignored
        # here because it cannot be fixed within cudf.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", (DeprecationWarning, FutureWarning)
            )
            if isinstance(left, pd.DataFrame):
                tm.assert_frame_equal(left, right, **kwargs)
            elif isinstance(left, pd.Series):
                tm.assert_series_equal(left, right, **kwargs)
            else:
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
        elif any(np.issubdtype(type(x), np.floating) for x in (left, right)):
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
        with pytest.raises(type(e) if check_exception_type else Exception):
            rfunc(*rfunc_args, **rfunc_kwargs)
    else:
        raise AssertionError("Expected to fail with an Exception.")


def _get_args_kwars_for_assert_exceptions(func_args_and_kwargs):
    if func_args_and_kwargs is None:
        return [], {}
    else:
        if len(func_args_and_kwargs) == 1:
            func_args, func_kwargs = [], {}
            if isinstance(func_args_and_kwargs[0], abc.Sequence):
                func_args = func_args_and_kwargs[0]
            elif isinstance(func_args_and_kwargs[0], abc.Mapping):
                func_kwargs = func_args_and_kwargs[0]
            else:
                raise ValueError(
                    "length 1 func_args_and_kwargs must be "
                    "either a Sequence or a Mapping"
                )
        elif len(func_args_and_kwargs) == 2:
            if not isinstance(func_args_and_kwargs[0], abc.Sequence):
                raise ValueError(
                    "Positional argument at 1st position of "
                    "func_args_and_kwargs should be a sequence."
                )
            if not isinstance(func_args_and_kwargs[1], abc.Mapping):
                raise ValueError(
                    "Key-word argument at 2nd position of "
                    "func_args_and_kwargs should be a dictionary mapping."
                )

            func_args, func_kwargs = func_args_and_kwargs
        else:
            raise ValueError("func_args_and_kwargs must be of length 1 or 2")
        return func_args, func_kwargs


def gen_rand(dtype, size, **kwargs):
    dtype = cudf.dtype(dtype)
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
            int(1e18) / _unit_to_nanoseconds_conversion[time_unit],
        )
        return pd.to_datetime(
            np.random.randint(low=low, high=high, size=size), unit=time_unit
        )
    elif dtype.kind in ("O", "U"):
        low = kwargs.get("low", 10)
        high = kwargs.get("high", 11)
        nchars = np.random.randint(low=low, high=high, size=1)[0]
        char_options = np.array(list(string.ascii_letters + string.digits))
        all_chars = "".join(np.random.choice(char_options, nchars * size))
        return np.array(
            [all_chars[nchars * i : nchars * (i + 1)] for i in range(size)]
        )

    raise NotImplementedError(f"dtype.kind={dtype.kind}")


def gen_rand_series(dtype, size, **kwargs):
    values = gen_rand(dtype, size, **kwargs)
    if kwargs.get("has_nulls", False):
        return cudf.Series.from_masked_array(values, random_bitmask(size))

    return cudf.Series(values)


def _decimal_series(input, dtype):
    return cudf.Series(
        [x if x is None else Decimal(x) for x in input],
        dtype=dtype,
    )


@contextmanager
def does_not_raise():
    yield


def assert_column_memory_eq(
    lhs: cudf.core.column.ColumnBase, rhs: cudf.core.column.ColumnBase
):
    """Assert the memory location and size of `lhs` and `rhs` are equivalent.

    Both data pointer and mask pointer are checked. Also recursively check for
    children to the same constraints. Also fails check if the number of
    children mismatches at any level.
    """

    def get_ptr(x) -> int:
        return x.get_ptr(mode="read") if x else 0

    assert get_ptr(lhs.base_data) == get_ptr(rhs.base_data)
    assert get_ptr(lhs.base_mask) == get_ptr(rhs.base_mask)
    assert lhs.base_size == rhs.base_size
    assert lhs.offset == rhs.offset
    assert lhs.size == rhs.size
    assert len(lhs.base_children) == len(rhs.base_children)
    for lhs_child, rhs_child in zip(lhs.base_children, rhs.base_children):
        assert_column_memory_eq(lhs_child, rhs_child)
    if isinstance(lhs, cudf.core.column.CategoricalColumn) and isinstance(
        rhs, cudf.core.column.CategoricalColumn
    ):
        assert_column_memory_eq(lhs.categories, rhs.categories)
        assert_column_memory_eq(lhs.codes, rhs.codes)


def assert_column_memory_ne(
    lhs: cudf.core.column.ColumnBase, rhs: cudf.core.column.ColumnBase
):
    try:
        assert_column_memory_eq(lhs, rhs)
    except AssertionError:
        return
    raise AssertionError("lhs and rhs holds the same memory.")


def _create_pandas_series_float64_default(
    data=None, index=None, dtype=None, *args, **kwargs
):
    # Wrapper around pd.Series using a float64
    # default dtype for empty data to silence warnings.
    # TODO: Remove this in pandas-2.0 upgrade
    if dtype is None and (
        data is None or (not is_scalar(data) and len(data) == 0)
    ):
        dtype = "float64"
    return pd.Series(data=data, index=index, dtype=dtype, *args, **kwargs)


def _create_cudf_series_float64_default(
    data=None, index=None, dtype=None, *args, **kwargs
):
    # Wrapper around cudf.Series using a float64
    # default dtype for empty data to silence warnings.
    # TODO: Remove this in pandas-2.0 upgrade
    if dtype is None and (
        data is None or (not is_scalar(data) and len(data) == 0)
    ):
        dtype = "float64"
    return cudf.Series(data=data, index=index, dtype=dtype, *args, **kwargs)


parametrize_numeric_dtypes_pairwise = pytest.mark.parametrize(
    "left_dtype,right_dtype",
    list(itertools.combinations_with_replacement(NUMERIC_TYPES, 2)),
)


@contextmanager
def expect_warning_if(condition, warning=FutureWarning, *args, **kwargs):
    """Catch a warning using pytest.warns if the expect_warning is True.

    All arguments are forwarded to pytest.warns if expect_warning is True.
    """
    if condition:
        with pytest.warns(warning, *args, **kwargs):
            yield
    else:
        yield


def sv_to_udf_str(sv):
    """
    Cast a string_view object to a udf_string object

    This placeholder function never runs in python
    It exists only for numba to have something to replace
    with the typing and lowering code below

    This is similar conceptually to needing a translation
    engine to emit an expression in target language "B" when
    there is no equivalent in the source language "A" to
    translate from. This function effectively defines the
    expression in language "A" and the associated typing
    and lowering describe the translation process, despite
    the expression having no meaning in language "A"
    """
    pass


@cuda_decl_registry.register_global(sv_to_udf_str)
class StringViewToUDFStringDecl(AbstractTemplate):
    def generic(self, args, kws):
        if isinstance(args[0], StringView) and len(args) == 1:
            return nb_signature(udf_string, string_view)


@cuda_lower(sv_to_udf_str, string_view)
def sv_to_udf_str_testing_lowering(context, builder, sig, args):
    return cast_string_view_to_udf_string(
        context, builder, sig.args[0], sig.return_type, args[0]
    )
