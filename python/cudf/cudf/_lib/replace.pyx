# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.api.types import is_scalar
from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view, mutable_column_view
from cudf._lib.cpp.replace cimport (
    clamp as cpp_clamp,
    find_and_replace_all as cpp_find_and_replace_all,
    normalize_nans_and_zeros as cpp_normalize_nans_and_zeros,
    replace_nulls as cpp_replace_nulls,
    replace_policy as cpp_replace_policy,
)
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def replace(Column input_col, Column values_to_replace,
            Column replacement_values):
    """
    Replaces values from values_to_replace with corresponding value from
    replacement_values in input_col

    Parameters
    ----------
    input_col : Column whose value will be updated
    values_to_replace : Column with values which needs to be replaced
    replacement_values : Column with values which will replace
    """

    cdef column_view input_col_view = input_col.view()
    cdef column_view values_to_replace_view = values_to_replace.view()
    cdef column_view replacement_values_view = replacement_values.view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_find_and_replace_all(input_col_view,
                                                 values_to_replace_view,
                                                 replacement_values_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def replace_nulls_column(Column input_col, Column replacement_values):
    """
    Replaces null values in input_col with corresponding values from
    replacement_values

    Parameters
    ----------
    input_col : Column whose value will be updated
    replacement_values : Column with values which will replace nulls
    """

    cdef column_view input_col_view = input_col.view()
    cdef column_view replacement_values_view = replacement_values.view()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_replace_nulls(input_col_view,
                                          replacement_values_view))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def replace_nulls_scalar(Column input_col, DeviceScalar replacement_value):
    """
    Replaces null values in input_col with replacement_value

    Parameters
    ----------
    input_col : Column whose value will be updated
    replacement_value : DeviceScalar with value which will replace nulls
    """

    cdef column_view input_col_view = input_col.view()
    cdef const scalar* replacement_value_scalar = replacement_value\
        .get_raw_ptr()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_replace_nulls(input_col_view,
                                          replacement_value_scalar[0]))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def replace_nulls_fill(Column input_col, object method):
    """
    Replaces null values in input_col with replacement_value

    Parameters
    ----------
    input_col : Column whose value will be updated
    method : 'ffill' or 'bfill'
    """

    cdef column_view input_col_view = input_col.view()

    cdef unique_ptr[column] c_result
    cdef cpp_replace_policy policy = (
        cpp_replace_policy.PRECEDING
        if method == 'ffill'
        else cpp_replace_policy.FOLLOWING
    )

    with nogil:
        c_result = move(cpp_replace_nulls(input_col_view, policy))

    return Column.from_unique_ptr(move(c_result))


def replace_nulls(
    Column input_col,
    object replacement=None,
    object method=None,
    object dtype=None
):
    """
    Calls one of the version of replace_nulls depending on type
    of replacement
    """

    if replacement is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")

    if replacement and method:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    if method:
        return replace_nulls_fill(input_col, method)
    elif is_scalar(replacement):
        return replace_nulls_scalar(
            input_col,
            as_device_scalar(replacement, dtype=dtype)
        )
    else:
        return replace_nulls_column(input_col, replacement)


@acquire_spill_lock()
def clamp(Column input_col, DeviceScalar lo, DeviceScalar lo_replace,
          DeviceScalar hi, DeviceScalar hi_replace):
    """
    Clip the input_col such that values < lo will be replaced by lo_replace
    and > hi will be replaced by hi_replace

    Parameters
    ----------
    input_col : Column whose value will be updated
    lo : DeviceScalar value for clipping lower values
    lo_replace : DeviceScalar value which will replace clipped with lo
    hi : DeviceScalar value for clipping upper values
    lo_replace : DeviceScalar value which will replace clipped with hi
    """

    cdef column_view input_col_view = input_col.view()
    cdef const scalar* lo_value = lo.get_raw_ptr()
    cdef const scalar* lo_replace_value = lo_replace.get_raw_ptr()
    cdef const scalar* hi_value = hi.get_raw_ptr()
    cdef const scalar* hi_replace_value = hi_replace.get_raw_ptr()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_clamp(
            input_col_view, lo_value[0],
            lo_replace_value[0], hi_value[0], hi_replace_value[0]))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def clamp(Column input_col, DeviceScalar lo, DeviceScalar hi):
    """
    Clip the input_col such that values < lo will be replaced by lo
    and > hi will be replaced by hi

    Parameters
    ----------
    input_col : Column whose value will be updated
    lo : DeviceScalar value for clipping lower values
    hi : DeviceScalar value for clipping upper values
    """

    cdef column_view input_col_view = input_col.view()
    cdef const scalar* lo_value = lo.get_raw_ptr()
    cdef const scalar* hi_value = hi.get_raw_ptr()

    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_clamp(input_col_view, lo_value[0], hi_value[0]))

    return Column.from_unique_ptr(move(c_result))


@acquire_spill_lock()
def clip(Column input_col, object lo, object hi):
    """
    Clip the input_col such that values < lo will be replaced by lo
    and > hi will be replaced by hi
    """

    lo_scalar = as_device_scalar(lo, dtype=input_col.dtype)
    hi_scalar = as_device_scalar(hi, dtype=input_col.dtype)

    return clamp(input_col, lo_scalar, hi_scalar)


@acquire_spill_lock()
def normalize_nans_and_zeros_inplace(Column input_col):
    """
    Inplace normalizing
    """

    cdef mutable_column_view input_col_view = input_col.mutable_view()
    with nogil:
        cpp_normalize_nans_and_zeros(input_col_view)


@acquire_spill_lock()
def normalize_nans_and_zeros_column(Column input_col):
    """
    Returns a new  normalized Column
    """

    cdef column_view input_col_view = input_col.view()
    cdef unique_ptr[column] c_result
    with nogil:
        c_result = move(cpp_normalize_nans_and_zeros(input_col_view))

    return Column.from_unique_ptr(move(c_result))


def normalize_nans_and_zeros(Column input_col, in_place=False):
    """
    Normalize the NaN and zeros in input_col
    Convert  -NaN  -> NaN
    Convert  -0.0  -> 0.0

    Parameters
    ----------
    input_col : Column that needs to be normalized
    in_place : boolean whether to normalize in place or return new column
    """

    if in_place is True:
        normalize_nans_and_zeros_inplace(input_col)
    else:
        return normalize_nans_and_zeros_column(input_col)
