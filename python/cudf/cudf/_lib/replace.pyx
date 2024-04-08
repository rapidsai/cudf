# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.api.types import is_scalar
from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar

from cudf._lib import pylibcudf
from cudf._lib.scalar import as_device_scalar


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

    return Column.from_pylibcudf(
        pylibcudf.replace.find_and_replace_all(
            input_col.to_pylibcudf(mode="read"),
            values_to_replace.to_pylibcudf(mode="read"),
            replacement_values.to_pylibcudf(mode="read"),
        )
    )


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
    return Column.from_pylibcudf(
        pylibcudf.replace.replace_nulls(
            input_col.to_pylibcudf(mode="read"),
            replacement_values.to_pylibcudf(mode="read"),
        )
    )


@acquire_spill_lock()
def replace_nulls_scalar(Column input_col, DeviceScalar replacement_value):
    """
    Replaces null values in input_col with replacement_value

    Parameters
    ----------
    input_col : Column whose value will be updated
    replacement_value : DeviceScalar with value which will replace nulls
    """
    return Column.from_pylibcudf(
        pylibcudf.replace.replace_nulls(
            input_col.to_pylibcudf(mode="read"),
            replacement_value.c_value,
        )
    )


@acquire_spill_lock()
def replace_nulls_fill(Column input_col, object method):
    """
    Replaces null values in input_col with replacement_value

    Parameters
    ----------
    input_col : Column whose value will be updated
    method : 'ffill' or 'bfill'
    """
    return Column.from_pylibcudf(
        pylibcudf.replace.replace_nulls(
            input_col.to_pylibcudf(mode="read"),
            pylibcudf.replace.ReplacePolicy.PRECEDING
            if method == 'ffill'
            else pylibcudf.replace.ReplacePolicy.FOLLOWING,
        )
    )


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
    return Column.from_pylibcudf(
        pylibcudf.replace.clamp(
            input_col.to_pylibcudf(mode="read"),
            lo.c_value,
            hi.c_value,
        )
    )


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
    pylibcudf.replace.normalize_nans_and_zeros(
        input_col.to_pylibcudf(mode="write"), inplace=True
    )


@acquire_spill_lock()
def normalize_nans_and_zeros_column(Column input_col):
    """
    Returns a new  normalized Column
    """
    return Column.from_pylibcudf(
        pylibcudf.replace.normalize_nans_and_zeros(
            input_col.to_pylibcudf(mode="read")
        )
    )


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
