# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

import cudf
from cudf.utils.dtypes import CUDF_STRING_DTYPE

if TYPE_CHECKING:
    from cudf._typing import DtypeObj


def is_dtype_obj_string(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF string type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF string type.
    """
    return (
        obj == CUDF_STRING_DTYPE
        or isinstance(obj, pd.StringDtype)
        or (
            isinstance(obj, pd.ArrowDtype)
            and (
                pa.types.is_string(obj.pyarrow_dtype)
                or pa.types.is_large_string(obj.pyarrow_dtype)
            )
        )
    )


def is_dtype_obj_list(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF list type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF list type.
    """
    return isinstance(obj, cudf.ListDtype) or (
        isinstance(obj, pd.ArrowDtype) and pa.types.is_list(obj.pyarrow_dtype)
    )


def is_dtype_obj_struct(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF struct type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF struct type.
    """
    return isinstance(obj, cudf.StructDtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_struct(obj.pyarrow_dtype)
    )


def is_dtype_obj_interval(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF interval type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF interval type.
    """
    return isinstance(obj, cudf.IntervalDtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_interval(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal type.
    """
    return (
        is_dtype_obj_decimal32(obj)
        or is_dtype_obj_decimal64(obj)
        or is_dtype_obj_decimal128(obj)
    )


def is_dtype_obj_decimal32(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal32 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal32 type.
    """
    return isinstance(obj, cudf.Decimal32Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal32(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal64(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal64 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal64 type.
    """
    return isinstance(obj, cudf.Decimal64Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal64(obj.pyarrow_dtype)
    )


def is_dtype_obj_decimal128(obj: DtypeObj) -> bool:
    """Check whether the provided dtype object is a cuDF decimal128 type.

    Parameters
    ----------
    obj : DtypeObj
        The dtype object to check.

    Returns
    -------
    bool
        Whether or not the dtype object is a cuDF decimal128 type.
    """
    return isinstance(obj, cudf.Decimal128Dtype) or (
        isinstance(obj, pd.ArrowDtype)
        and pa.types.is_decimal128(obj.pyarrow_dtype)
    )


def is_dtype_obj_numeric(
    dtype: DtypeObj, include_decimal: bool = True
) -> bool:
    """
    Check whether the provided dtype object is a numeric type.

    Parameters
    ----------
    dtype: DtypeObj
        The dtype object to check.
    include_decimal: bool, default True
        Whether to include decimal types in the check.

    Returns
    -------
    bool
        Whether or not the dtype object is a numeric type.
    """
    is_non_decimal = dtype.kind in set("iufb")
    if include_decimal:
        return is_non_decimal or is_dtype_obj_decimal(dtype)
    else:
        return is_non_decimal
