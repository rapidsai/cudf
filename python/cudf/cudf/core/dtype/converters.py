# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import zoneinfo
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa

from cudf.core.dtypes import (
    CategoricalDtype,
    Decimal32Dtype,
    Decimal64Dtype,
    Decimal128Dtype,
    IntervalDtype,
    ListDtype,
    StructDtype,
)

if TYPE_CHECKING:
    from cudf._typing import DtypeObj


# TODO: Move this to dtype/validators.py
def is_default_cudf_dtype(dtype_to_check: Any) -> bool:
    from cudf.utils.dtypes import CUDF_STRING_DTYPE

    if isinstance(dtype_to_check, np.dtype):
        return (
            dtype_to_check.kind in set("iufb")
            or dtype_to_check == CUDF_STRING_DTYPE
        )
    elif isinstance(dtype_to_check, CategoricalDtype):
        return is_default_cudf_dtype(dtype_to_check.categories.dtype)
    elif isinstance(dtype_to_check, IntervalDtype):
        return is_default_cudf_dtype(dtype_to_check.subtype)
    elif isinstance(dtype_to_check, StructDtype):
        return all(
            is_default_cudf_dtype(field)
            for field in dtype_to_check.fields.values()
        )
    elif isinstance(dtype_to_check, ListDtype):
        return is_default_cudf_dtype(dtype_to_check.element_type)
    else:
        return isinstance(
            dtype_to_check,
            (
                pd.DatetimeTZDtype,
                Decimal128Dtype,
                Decimal64Dtype,
                Decimal32Dtype,
            ),
        )


@functools.singledispatch
@functools.cache
def _convert_to_arrowdtype(dtype: DtypeObj) -> pd.ArrowDtype:
    raise TypeError(f"Cannot convert {dtype} to a pandas.ArrowDtype")


@_convert_to_arrowdtype.register(pd.ArrowDtype)
def _(dtype) -> pd.ArrowDtype:
    return dtype


@_convert_to_arrowdtype.register(pd.BooleanDtype)
@_convert_to_arrowdtype.register(pd.Int8Dtype)
@_convert_to_arrowdtype.register(pd.Int16Dtype)
@_convert_to_arrowdtype.register(pd.Int32Dtype)
@_convert_to_arrowdtype.register(pd.Int64Dtype)
@_convert_to_arrowdtype.register(pd.UInt8Dtype)
@_convert_to_arrowdtype.register(pd.UInt16Dtype)
@_convert_to_arrowdtype.register(pd.UInt32Dtype)
@_convert_to_arrowdtype.register(pd.UInt64Dtype)
@_convert_to_arrowdtype.register(pd.Float32Dtype)
def _(dtype) -> pd.ArrowDtype:
    return pd.ArrowDtype(pa.from_numpy_dtype(dtype.numpy_dtype))


@_convert_to_arrowdtype.register(pd.StringDtype)
def _(dtype) -> pd.ArrowDtype:
    return pd.ArrowDtype(pa.large_string())


@_convert_to_arrowdtype.register(pd.DatetimeTZDtype)
def _(dtype) -> pd.ArrowDtype:
    return pd.ArrowDtype(pa.timestamp(dtype.unit, str(dtype.tz)))


@_convert_to_arrowdtype.register(ListDtype)
@_convert_to_arrowdtype.register(StructDtype)
@_convert_to_arrowdtype.register(IntervalDtype)
@_convert_to_arrowdtype.register(Decimal128Dtype)
@_convert_to_arrowdtype.register(Decimal64Dtype)
@_convert_to_arrowdtype.register(Decimal32Dtype)
def _(dtype) -> pd.ArrowDtype:
    return pd.ArrowDtype(dtype.to_arrow())


@_convert_to_arrowdtype.register(np.dtype)
def _(dtype) -> pd.ArrowDtype:
    from cudf.utils.dtypes import CUDF_STRING_DTYPE

    if dtype.kind in {"i", "u", "f", "b", "m", "M"}:
        return pd.ArrowDtype(pa.from_numpy_dtype(dtype))
    elif dtype == CUDF_STRING_DTYPE:
        return pd.ArrowDtype(pa.large_string())
    else:
        raise TypeError(f"Cannot convert {dtype} to a pandas.ArrowDtype")


@functools.singledispatch
@functools.cache
def _convert_to_pandas_nullable_extension_type(dtype: DtypeObj) -> DtypeObj:
    raise TypeError(
        f"Cannot convert {dtype} to a pandas nullable extension type"
    )


@_convert_to_pandas_nullable_extension_type.register(pd.BooleanDtype)
@_convert_to_pandas_nullable_extension_type.register(pd.Int16Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.Int32Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.Int64Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.UInt8Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.UInt16Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.UInt32Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.UInt64Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.Float32Dtype)
@_convert_to_pandas_nullable_extension_type.register(pd.StringDtype)
def _(dtype) -> DtypeObj:
    return dtype


@_convert_to_pandas_nullable_extension_type.register(pd.ArrowDtype)
def _(dtype) -> DtypeObj:
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        return pd.BooleanDtype()
    elif pa.types.is_int8(pa_dtype):
        return pd.Int8Dtype()
    elif pa.types.is_int16(pa_dtype):
        return pd.Int16Dtype()
    elif pa.types.is_int32(pa_dtype):
        return pd.Int32Dtype()
    elif pa.types.is_int64(pa_dtype):
        return pd.Int64Dtype()
    elif pa.types.is_uint8(pa_dtype):
        return pd.UInt8Dtype()
    elif pa.types.is_uint16(pa_dtype):
        return pd.UInt16Dtype()
    elif pa.types.is_uint32(pa_dtype):
        return pd.UInt32Dtype()
    elif pa.types.is_uint64(pa_dtype):
        return pd.UInt64Dtype()
    elif pa.types.is_float32(pa_dtype):
        return pd.Float32Dtype()
    elif pa.types.is_float64(pa_dtype):
        return pd.Float64Dtype()
    elif pa.types.is_string(pa_dtype) or pa.types.is_large_string(pa_dtype):
        # TODO(pandas3.0): Should this use na_value=pd.NA or np.nan?
        return pd.StringDtype(na_value=pd.NA)
    else:
        raise TypeError(
            f"Cannot convert {dtype} to a pandas nullable extension type"
        )


@_convert_to_pandas_nullable_extension_type.register(np.dtype)
def _(dtype) -> DtypeObj:
    from cudf.utils.dtypes import CUDF_STRING_DTYPE

    np_type = dtype.type
    if np_type == np.bool_:
        return pd.BooleanDtype()
    if dtype == np.int8:
        return pd.Int8Dtype()
    elif dtype == np.int16:
        return pd.Int16Dtype()
    elif dtype == np.int32:
        return pd.Int32Dtype()
    elif dtype == np.int64:
        return pd.Int64Dtype()
    elif dtype == np.uint8:
        return pd.UInt8Dtype()
    elif dtype == np.uint16:
        return pd.UInt16Dtype()
    elif dtype == np.uint32:
        return pd.UInt32Dtype()
    elif dtype == np.uint64:
        return pd.UInt64Dtype()
    elif dtype == np.float32:
        return pd.Float32Dtype()
    elif dtype == np.float64:
        return pd.Float64Dtype()
    elif dtype == CUDF_STRING_DTYPE:
        return pd.StringDtype(na_value=np.nan)
    else:
        raise TypeError(
            f"Cannot convert {dtype} to a pandas nullable extension type"
        )


@functools.singledispatch
# TODO: Cannot cache since CategoricalDtype is not hashable
# https://github.com/rapidsai/cudf/issues/14027
# @functools.cache
def _convert_to_default_cudf_dtype(dtype: DtypeObj) -> DtypeObj:
    raise TypeError(f"Cannot convert {dtype} to a default cudf dtype")


@_convert_to_default_cudf_dtype.register(pd.DatetimeTZDtype)
@_convert_to_default_cudf_dtype.register(Decimal128Dtype)
@_convert_to_default_cudf_dtype.register(Decimal64Dtype)
@_convert_to_default_cudf_dtype.register(Decimal32Dtype)
@_convert_to_default_cudf_dtype.register(ListDtype)
@_convert_to_default_cudf_dtype.register(StructDtype)
@_convert_to_default_cudf_dtype.register(IntervalDtype)
@_convert_to_default_cudf_dtype.register(CategoricalDtype)
def _(dtype) -> DtypeObj:
    return dtype


@_convert_to_default_cudf_dtype.register(np.dtype)
def _(dtype) -> DtypeObj:
    from cudf.utils.dtypes import CUDF_STRING_DTYPE

    if (
        dtype.kind in {"i", "u", "f", "b", "m", "M"}
        or dtype == CUDF_STRING_DTYPE
    ):
        return dtype
    else:
        raise TypeError(f"Cannot convert {dtype} to a default cudf dtype")


@_convert_to_default_cudf_dtype.register(pd.ArrowDtype)
def _(dtype) -> DtypeObj:
    pa_dtype = dtype.pyarrow_dtype
    if dtype.kind in {"i", "u", "f", "b", "m"} or (
        dtype.kind == "M" and dtype.tz is None
    ):
        return pa_dtype.to_pandas_dtype()
    elif dtype.kind == "M" and dtype.tz is not None:
        # pyarrow returns a pytz object, but cuDF expects a zoneinfo object
        return pd.DatetimeTZDtype(
            pa_dtype.unit, zoneinfo.ZoneInfo(pa_dtype.tz)
        )
    elif pa.types.is_string(
        pa_dtype.pyarrow_dtype
    ) or pa.types.is_large_string(pa_dtype.pyarrow_dtype):
        from cudf.utils.dtypes import CUDF_STRING_DTYPE

        return CUDF_STRING_DTYPE
    elif pa.types.is_decimal128(pa_dtype):
        return Decimal128Dtype.from_arrow(pa_dtype)
    elif pa.types.is_decimal64(pa_dtype):
        return Decimal64Dtype.from_arrow(pa_dtype)
    elif pa.types.is_decimal32(pa_dtype):
        return Decimal32Dtype.from_arrow(pa_dtype)
    elif pa.types.is_list(pa_dtype):
        return ListDtype.from_arrow(pa_dtype)
    elif pa.types.is_struct(pa_dtype):
        return StructDtype.from_arrow(pa_dtype)
    elif pa.types.is_interval(pa_dtype):
        return IntervalDtype.from_arrow(pa_dtype)
    else:
        raise TypeError(f"Cannot convert {dtype} to a default cudf dtype")


def get_dtype_of_same_variant(
    variant_dtype: DtypeObj, target_dtype: DtypeObj
) -> DtypeObj:
    """
    Return a dtype that is equivalent to target_dtype but as the same variant as variant_dtype.

    Parameters
    ----------
    variant_dtype : DtypeObj
        A reference dtype to inform which type variant to return
        (e.g. the default type system, pandas nullable extension types, pandas.ArrowDtype)
    target_dtype : DtypeObj
        A reference dtype to inform the desired return type.

    Returns
    -------
    DtypeObj

    Examples
    --------
    >>> get_dtype_of_same_variant(np.dtype(np.float64), pd.BooleanDtype())
    dtype('bool')
    >>> get_dtype_of_same_variant(pd.BooleanDtype(), np.dtype(np.float64))
    Float64Dtype()
    >>> get_dtype_of_same_variant(pd.ArrowDtype(pa.float64()), pd.BooleanDtype())
    bool[pyarrow]  # ArrowDtype
    """
    from cudf.utils.dtypes import is_pandas_nullable_extension_dtype

    if isinstance(variant_dtype, pd.ArrowDtype):
        converter = _convert_to_arrowdtype
    elif is_pandas_nullable_extension_dtype(variant_dtype):
        converter = _convert_to_pandas_nullable_extension_type
    elif is_default_cudf_dtype(variant_dtype):
        converter = _convert_to_default_cudf_dtype
    else:
        raise ValueError(
            f"The variant type should be a default cudf dtype, a pandas nullable extension dtype, or a pandas.ArrowDtype."
            f"Unsupported variant dtype: {variant_dtype}"
        )
    # TOOO: May need to wrap this in a try/except:TypeError to mirror the old behavior
    return converter(target_dtype)
