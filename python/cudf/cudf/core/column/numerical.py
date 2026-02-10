# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Self, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.buffer import as_buffer
from cudf.core.column.categorical import CategoricalColumn
from cudf.core.column.column import (
    ColumnBase,
    as_column,
    column_empty,
)
from cudf.core.column.numerical_base import NumericalBaseColumn
from cudf.core.column.utils import access_columns
from cudf.core.dtypes import CategoricalDtype
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    CUDF_STRING_DTYPE,
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    find_common_type,
    get_dtype_of_same_kind,
    get_dtype_of_same_type,
    is_pandas_nullable_extension_dtype,
    min_signed_type,
    min_unsigned_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column import DecimalBaseColumn
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.string import StringColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.dtypes import DecimalDtype


class NumericalColumn(NumericalBaseColumn):
    """A Column object for Numeric types."""

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS
    _VALID_PLC_TYPES = {
        plc.TypeId.INT8,
        plc.TypeId.INT16,
        plc.TypeId.INT32,
        plc.TypeId.INT64,
        plc.TypeId.UINT8,
        plc.TypeId.UINT16,
        plc.TypeId.UINT32,
        plc.TypeId.UINT64,
        plc.TypeId.FLOAT32,
        plc.TypeId.FLOAT64,
        plc.TypeId.BOOL8,
    }

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """Float columns return np.nan as NA value in pandas compatibility mode."""
        if (
            cudf.get_option("mode.pandas_compatible")
            and self.dtype.kind == "f"
            and not is_pandas_nullable_extension_dtype(self.dtype)
        ):
            return np.nan
        return super()._PANDAS_NA_VALUE

    @classmethod
    def _validate_args(
        cls, plc_column: plc.Column, dtype: np.dtype
    ) -> tuple[plc.Column, np.dtype]:
        plc_column, dtype = super()._validate_args(plc_column, dtype)
        if (
            cudf.get_option("mode.pandas_compatible")
            and dtype.kind not in "iufb"
        ) or (
            not cudf.get_option("mode.pandas_compatible")
            and not (isinstance(dtype, np.dtype) and dtype.kind in "iufb")
        ):
            raise ValueError(
                f"dtype must be a floating, integer or boolean dtype. Got: {dtype}"
            )
        return plc_column, dtype

    def __contains__(self, item: ScalarLike) -> bool:
        """
        Returns True if column contains item, else False.
        """
        # Handles improper item types
        # Fails if item is of type None, so the handler.
        try:
            search_item = self.dtype.type(item)
            if search_item != item and self.dtype.kind != "f":
                return False
        except (TypeError, ValueError):
            return False
        # TODO: Use `scalar`-based `contains` wrapper
        return self.contains(
            as_column(
                [search_item],
                dtype=self.dtype,
                nan_as_null=not cudf.get_option("mode.pandas_compatible"),
            ),
        ).any()

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the NumericalColumn.
        """
        dtype = self.dtype
        if is_pandas_nullable_extension_dtype(dtype):
            dtype = getattr(dtype, "numpy_dtype", dtype)

        if len(self) == 0:
            return cp.empty(0, dtype=dtype)

        col = self
        if col.has_nulls():
            if dtype.kind == "b":
                raise ValueError(
                    f"Column must have no nulls for dtype={col.dtype}"
                )
            elif dtype.kind != "f":
                dtype = np.dtype(np.float64)
                col = col.astype(dtype)  # type: ignore[assignment]
            col = col.fillna(np.nan)

        return cp.asarray(col).view(dtype)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        if isinstance(value, (bool, np.bool_)) and self.dtype.kind != "b":
            raise ValueError(
                f"Cannot use a {type(value).__name__} to find an index of "
                f"a {self.dtype} Index."
            )
        elif (
            self.dtype.kind in {"c", "f"}
            and isinstance(value, (float, np.floating))
            and np.isnan(value)
        ):
            return self.isnan().indices_of(True)
        else:
            return super().indices_of(value)

    def has_nulls(self, include_nan: bool = False) -> bool:
        return bool(self.null_count != 0) or (
            include_nan and bool(self.nan_count != 0)
        )

    def isnan(self) -> ColumnBase:
        """Identify NaN values in a Column.

        Only meaningful for float dtypes. For integer and boolean columns,
        returns a column of False values.
        """
        if self.dtype.kind != "f":
            return as_column(False, length=len(self))
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.unary.is_nan(self.plc_column), np.dtype(np.bool_)
            )

    def notnan(self) -> ColumnBase:
        """Identify non-NaN values in a Column.

        Only meaningful for float dtypes. For integer and boolean columns,
        returns a column of True values.
        """
        if self.dtype.kind != "f":
            return as_column(True, length=len(self))
        with self.access(mode="read", scope="internal"):
            return ColumnBase.create(
                plc.unary.is_not_nan(self.plc_column), np.dtype(np.bool_)
            )

    def isnull(self) -> ColumnBase:
        """Identify missing values in a Column.

        For float columns, NaN values are also considered null.
        """
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            return as_column(False, length=len(self))

        with self.access(mode="read", scope="internal"):
            is_null_plc = plc.unary.is_null(self.plc_column)
            if self.dtype.kind == "f":
                is_nan_plc = plc.unary.is_nan(self.plc_column)
                result_plc = plc.binaryop.binary_operation(
                    is_null_plc,
                    is_nan_plc,
                    plc.binaryop.BinaryOperator.BITWISE_OR,
                    plc.types.DataType(plc.types.TypeId.BOOL8),
                )
                return ColumnBase.create(result_plc, np.dtype(np.bool_))
            return ColumnBase.create(is_null_plc, np.dtype(np.bool_))

    def notnull(self) -> ColumnBase:
        """Identify non-missing values in a Column.

        For float columns, NaN values are considered null and excluded.
        """
        if not self.has_nulls(include_nan=self.dtype.kind == "f"):
            result = as_column(True, length=len(self))
        else:
            with self.access(mode="read", scope="internal"):
                is_valid_plc = plc.unary.is_valid(self.plc_column)
                if self.dtype.kind == "f":
                    is_not_nan_plc = plc.unary.is_not_nan(self.plc_column)
                    result_plc = plc.binaryop.binary_operation(
                        is_valid_plc,
                        is_not_nan_plc,
                        plc.binaryop.BinaryOperator.BITWISE_AND,
                        plc.types.DataType(plc.types.TypeId.BOOL8),
                    )
                    result = ColumnBase.create(result_plc, np.dtype(np.bool_))
                else:
                    result = ColumnBase.create(
                        is_valid_plc, np.dtype(np.bool_)
                    )

        return result

    def element_indexing(self, index: int) -> ScalarLike | None:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            return self.dtype.type(result.as_py())
        return result

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if is_scalar(value):
            if value is cudf.NA or value is None:
                scalar = pa.scalar(
                    None, type=cudf_dtype_to_pa_type(self.dtype)
                )
            else:
                try:
                    scalar = pa.scalar(value)
                except ValueError as err:
                    raise TypeError(
                        f"Cannot set value of type {type(value)} to column of type {self.dtype}"
                    ) from err
            is_scalar_bool = pa.types.is_boolean(scalar.type)
            if (is_scalar_bool and self.dtype.kind != "b") or (
                not is_scalar_bool and self.dtype.kind == "b"
            ):
                raise TypeError(
                    f"Invalid value {value} for dtype {self.dtype}"
                )
            return pa_scalar_to_plc_scalar(
                scalar.cast(cudf_dtype_to_pa_type(self.dtype))
            )
        else:
            col = as_column(value)
            if col.dtype.kind == "b" and self.dtype.kind != "b":
                raise TypeError(
                    f"Invalid value {value} for dtype {self.dtype}"
                )
            return col.astype(self.dtype)

    def __invert__(self) -> ColumnBase:
        if (dtype_kind := self.dtype.kind) in "ui":
            return self.unary_operator("invert")
        elif dtype_kind == "b":
            return self.unary_operator("not")
        else:
            return super().__invert__()

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        int_float_dtype_mapping = {
            np.int8: np.float32,
            np.int16: np.float32,
            np.int32: np.float32,
            np.int64: np.float64,
            np.uint8: np.float32,
            np.uint16: np.float32,
            np.uint32: np.float64,
            np.uint64: np.float64,
            np.bool_: np.float32,
        }
        if cudf.get_option("mode.pandas_compatible"):
            int_float_dtype_mapping = {
                np.int8: np.float64,
                np.int16: np.float64,
                np.int32: np.float64,
                np.int64: np.float64,
                np.uint8: np.float64,
                np.uint16: np.float64,
                np.uint32: np.float64,
                np.uint64: np.float64,
                np.bool_: np.float64,
            }

        cmp_ops = {
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
            "__eq__",
            "__ne__",
        }
        out_dtype = None
        if op in {"__truediv__", "__rtruediv__"}:
            # Division with integer types results in a suitable float.
            if truediv_type := int_float_dtype_mapping.get(
                self.dtype.numpy_dtype.type
                if is_pandas_nullable_extension_dtype(self.dtype)
                else self.dtype.type
            ):
                return self.astype(
                    get_dtype_of_same_kind(self.dtype, np.dtype(truediv_type))
                )._binaryop(other, op)
        elif op in cmp_ops:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))

            # If `other` is a Python integer and it is out-of-bounds
            # promotion could fail but we can trivially define the result
            # in terms of `notnull` or `NULL_NOT_EQUALS`.
            if type(other) is int and self.dtype.kind in "iu":
                truthiness = None
                iinfo = np.iinfo(self.dtype)
                if iinfo.min > other:
                    truthiness = op in {"__ne__", "__gt__", "__ge__"}
                elif iinfo.max < other:
                    truthiness = op in {"__ne__", "__lt__", "__le__"}

                # Compare with minimum value so that the result is true/false
                if truthiness is True:
                    other = iinfo.min
                    op = "__ge__"
                elif truthiness is False:
                    other = iinfo.min
                    op = "__lt__"

        elif op in {"NULL_EQUALS", "NULL_NOT_EQUALS"}:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))

        reflect, op = self._check_reflected_op(op)
        if (other := self._normalize_binop_operand(other)) is NotImplemented:
            return NotImplemented
        other_cudf_dtype = (
            cudf_dtype_from_pa_type(other.type)
            if isinstance(other, pa.Scalar)
            else other.dtype
        )

        if out_dtype is None:
            out_dtype = find_common_type((self.dtype, other_cudf_dtype))
            if op in {"__mod__", "__floordiv__"}:
                tmp = self if reflect else other
                tmp_dtype = self.dtype if reflect else other_cudf_dtype
                # Guard against division by zero for integers.
                if tmp_dtype.kind in "iu" and (
                    (isinstance(tmp, NumericalColumn) and 0 in tmp)
                    or (isinstance(tmp, pa.Scalar) and tmp.as_py() == 0)
                ):
                    out_dtype = get_dtype_of_same_kind(
                        out_dtype, np.dtype(np.float64)
                    )

        if op in {"__and__", "__or__", "__xor__"}:
            if self.dtype.kind == "f" or other_cudf_dtype.kind == "f":
                raise TypeError(
                    f"Operation 'bitwise {op[2:-2]}' not supported between "
                    f"{self.dtype.type.__name__} and "
                    f"{other_cudf_dtype.type.__name__}"
                )
            if self.dtype.kind == "b" and other_cudf_dtype.kind == "b":
                out_dtype = get_dtype_of_same_kind(
                    self.dtype, np.dtype(np.bool_)
                )
            elif self.dtype.kind == "b" or other_cudf_dtype.kind == "b":
                out_dtype = get_dtype_of_same_kind(
                    out_dtype, np.dtype(np.bool_)
                )

        elif (
            op == "__pow__"
            and self.dtype.kind in "iu"
            and (other_cudf_dtype.kind in "iu")
        ):
            op = "INT_POW"

        lhs_dtype, rhs_dtype = (
            (other_cudf_dtype, self.dtype)
            if reflect
            else (self.dtype, other_cudf_dtype)
        )
        lhs, rhs = (other, self) if reflect else (self, other)
        if out_dtype.kind == "f" and is_pandas_nullable_extension_dtype(
            out_dtype
        ):
            if (
                not is_pandas_nullable_extension_dtype(lhs_dtype)
                and lhs_dtype.kind == "f"
                and isinstance(lhs, NumericalColumn)
            ):
                lhs = lhs.nans_to_nulls()
            if (
                not is_pandas_nullable_extension_dtype(rhs_dtype)
                and rhs_dtype.kind == "f"
                and isinstance(rhs, NumericalColumn)
            ):
                rhs = rhs.nans_to_nulls()
        lhs_binaryop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(lhs) if isinstance(lhs, pa.Scalar) else lhs
        )
        rhs_binaryop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(rhs) if isinstance(rhs, pa.Scalar) else rhs
        )

        res = binaryop.binaryop(lhs_binaryop, rhs_binaryop, op, out_dtype)
        if (
            is_pandas_nullable_extension_dtype(out_dtype)
            and out_dtype.kind == "f"
        ):
            # If the output dtype is a pandas nullable extension type,
            # we need to ensure that the result is a NumericalColumn.
            res = res.nans_to_nulls()
        if op in {"__mod__", "__floordiv__"} and tmp_dtype.kind == "b":
            res = res.astype(
                get_dtype_of_same_kind(out_dtype, np.dtype(np.int8))
            )
        elif op == "INT_POW" and res.null_count:
            if (
                isinstance(lhs_binaryop, plc.Scalar)
                and lhs_binaryop.to_py() == 1
                and isinstance(rhs_binaryop, ColumnBase)
                and rhs_binaryop.null_count > 0
            ):
                res = res.fillna(lhs_binaryop.to_py())
        elif (
            cudf.get_option("mode.pandas_compatible")
            and op in cmp_ops
            and not is_pandas_nullable_extension_dtype(self.dtype)
        ):
            res = res.fillna(op == "__ne__")
        return res

    def nans_to_nulls(self: Self) -> Self:
        # Only floats can contain nan.
        if self.dtype.kind != "f" or self.nan_count == 0:
            return self
        with self.access(mode="read", scope="internal"):
            # When computing a null mask to set back to the column, since the column may
            # have been sliced and have an offset, we need to compute the mask of the
            # equivalent unsliced column so that the mask bits will be appropriately
            # shifted..
            shifted_column = plc.Column(
                self.plc_column.type(),
                self.plc_column.size() + self.plc_column.offset(),
                self.plc_column.data(),
                self.plc_column.null_mask(),
                self.plc_column.null_count(),
                0,
                self.plc_column.children(),
            )
            mask, null_count = plc.transform.nans_to_nulls(shifted_column)
            return self.set_mask(as_buffer(mask), null_count)

    def _normalize_binop_operand(self, other: Any) -> pa.Scalar | ColumnBase:
        if isinstance(other, ColumnBase):
            if not isinstance(other, type(self)):
                return NotImplemented
            return other
        # TODO: cupy scalars are just aliases for numpy scalars, so extracting a scalar
        # from a cupy array would always require a D2H copy. As a result, cupy does not
        # produce scalars without explicit casting requests
        # https://docs.cupy.dev/en/stable/user_guide/difference.html#zero-dimensional-array
        # The below logic for type inference relies on numpy, however, so we need to go
        # that route for now. If possible we should find a way to avoid this.
        if isinstance(other, cp.ndarray) and other.ndim == 0:
            other = cp.asnumpy(other)[()]
        elif isinstance(other, np.ndarray) and other.ndim == 0:
            other = other[()]

        if is_scalar(other):
            if is_na_like(other):
                if isinstance(
                    other, (np.datetime64, np.timedelta64)
                ) and np.isnat(other):
                    return NotImplemented
                return pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            if not isinstance(other, (int, float, complex)):
                # Go via NumPy to get the value
                other = np.array(other)
                if other.dtype.kind in "uifc":
                    other = other.item()

            # Try and match pandas and hence numpy. Deduce the common
            # dtype via the _value_ of other, and the dtype of self on NumPy 1.x
            # with NumPy 2, we force weak promotion even for our/NumPy scalars
            # to match pandas 2.2.
            # Weak promotion is not at all simple:
            # np.result_type(0, np.uint8)
            #   => np.uint8
            # np.result_type(np.asarray([0], dtype=np.int64), np.uint8)
            #   => np.int64
            # np.promote_types(np.int64(0), np.uint8)
            #   => np.int64
            # np.promote_types(np.asarray([0], dtype=np.int64).dtype, np.uint8)
            #   => np.int64
            if is_pandas_nullable_extension_dtype(self.dtype):
                if isinstance(self.dtype, pd.ArrowDtype):
                    common_dtype = cudf.utils.dtypes.find_common_type(
                        [self.dtype, other]
                    )
                else:
                    common_dtype = get_dtype_of_same_kind(
                        self.dtype,
                        np.result_type(self.dtype.numpy_dtype, other),  # noqa: TID251
                    )
            else:
                common_dtype = np.result_type(self.dtype, other)  # noqa: TID251
            if common_dtype.kind in {"b", "i", "u", "f"}:  # type: ignore[union-attr]
                if self.dtype.kind == "b" and not isinstance(other, bool):
                    common_dtype = min_signed_type(other)
                return pa.scalar(
                    other, type=cudf_dtype_to_pa_type(common_dtype)
                )
            else:
                return NotImplemented
        else:
            return NotImplemented

    def int2ip(self) -> StringColumn:
        if self.dtype != np.dtype(np.uint32):
            raise TypeError("Only uint32 type can be converted to ip")
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_ipv4.integers_to_ipv4(
                self.plc_column
            )
            return cast(
                cudf.core.column.string.StringColumn,
                ColumnBase.create(plc_column, CUDF_STRING_DTYPE),
            )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        col = self
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(dtype, np.dtype)
            and dtype.kind == "O"
        ):
            raise ValueError(
                "Cannot convert numerical column to string column "
                "when dtype is an object dtype in pandas compatibility mode."
            )
        if len(self) == 0:
            return cast(
                cudf.core.column.StringColumn,
                column_empty(0, dtype=CUDF_STRING_DTYPE),
            )

        conv_func: Callable[[plc.Column], plc.Column]
        if self.dtype.kind == "b":
            conv_func = functools.partial(
                plc.strings.convert.convert_booleans.from_booleans,
                true_string=pa_scalar_to_plc_scalar(pa.scalar("True")),
                false_string=pa_scalar_to_plc_scalar(pa.scalar("False")),
            )
        elif self.dtype.kind in {"i", "u"}:
            conv_func = plc.strings.convert.convert_integers.from_integers
        elif self.dtype.kind == "f":
            if cudf.get_option(
                "mode.pandas_compatible"
            ) and is_pandas_nullable_extension_dtype(dtype):
                # In pandas compatibility mode, we convert nans to nulls
                col = self.nans_to_nulls()
            conv_func = plc.strings.convert.convert_floats.from_floats
        else:
            raise ValueError(f"No string conversion from type {self.dtype}")

        with col.access(mode="read", scope="internal"):
            return cast(
                cudf.core.column.string.StringColumn,
                ColumnBase.create(conv_func(col.plc_column), dtype),
            )

    def _as_temporal_column(self, dtype: np.dtype) -> plc.Column:
        """Convert Self to a temporal pylibcudf Column for as_datetime_column and as_timedelta_column"""
        return plc.Column(
            data_type=dtype_to_pylibcudf_type(dtype),
            size=self.size,
            data=self.astype(np.dtype(np.int64)).data,
            mask=self.mask,
            null_count=self.null_count,
            offset=self.offset,
            children=[],
        )

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        return cast(
            cudf.core.column.datetime.DatetimeColumn,
            ColumnBase.create(self._as_temporal_column(dtype), dtype),
        )

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return cast(
            cudf.core.column.timedelta.TimeDeltaColumn,
            ColumnBase.create(self._as_temporal_column(dtype), dtype),
        )

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalBaseColumn:
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def as_numerical_column(self, dtype: DtypeObj) -> NumericalColumn:
        if dtype == self.dtype:
            return self

        if cudf.get_option("mode.pandas_compatible"):
            if (
                is_pandas_nullable_extension_dtype(self.dtype)
                and isinstance(dtype, np.dtype)
                and self.null_count > 0
            ):
                if dtype.kind in "iu":
                    raise ValueError("cannot convert NA to integer")
                elif dtype.kind == "b":
                    raise ValueError("cannot convert float NaN to bool")

            if (
                not is_pandas_nullable_extension_dtype(self.dtype)
                and is_pandas_nullable_extension_dtype(dtype)
                and dtype.kind == "f"
            ):
                res = self.nans_to_nulls().cast(dtype=dtype)
                res._dtype = dtype
                return res  # type: ignore[return-value]

            if (
                self.dtype.kind == "f"
                and dtype.kind == "b"
                and not is_pandas_nullable_extension_dtype(dtype)
                and self.has_nulls()
            ):
                return self.fillna(np.nan).cast(dtype=dtype)  # type: ignore[return-value]

            if dtype_to_pylibcudf_type(dtype) == dtype_to_pylibcudf_type(
                self.dtype
            ):
                # Short-circuit the cast if the dtypes are equivalent
                # but not the same type object.
                if (
                    is_pandas_nullable_extension_dtype(dtype)
                    and isinstance(self.dtype, np.dtype)
                    and self.dtype.kind == "f"
                ):
                    # If the dtype is a pandas nullable extension type, we need to
                    # float column doesn't have any NaNs.
                    res = self.nans_to_nulls()
                    res._dtype = dtype
                    return res
                else:
                    self._dtype = dtype
                    return self
            if self.dtype.kind == "f" and dtype.kind in "iu":
                if not is_pandas_nullable_extension_dtype(dtype) and (
                    self.nan_count > 0
                    or np.isinf(self.min())
                    or np.isinf(self.max())
                ):
                    raise TypeError(
                        "Cannot convert non-finite values (NA or inf) to integer"
                    )
                # If casting from float to int, we need to convert nans to nulls
                res = self.nans_to_nulls().cast(dtype=dtype)
                res._dtype = dtype
                return res  # type: ignore[return-value]

        return self.cast(dtype=dtype)  # type: ignore[return-value]

    @functools.cached_property
    def nan_count(self) -> int:
        if self.dtype.kind != "f":
            return super().nan_count
        return self.isnan().sum()

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
        try:
            lhs, rhs = super()._process_values_for_isin(values)
        except TypeError:
            # Can remove once dask 25.04 is the minimum version
            # https://github.com/dask/dask/pull/11869
            if isinstance(values, np.ndarray) and values.dtype.kind == "O":
                return super()._process_values_for_isin(values.tolist())
            else:
                raise
        if lhs.dtype != rhs.dtype and rhs.dtype != CUDF_STRING_DTYPE:
            if rhs.can_cast_safely(lhs.dtype):
                rhs = rhs.astype(lhs.dtype)
            elif lhs.can_cast_safely(rhs.dtype):
                lhs = lhs.astype(rhs.dtype)
        return lhs, rhs

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=True)

    def _min_column_type(self, expected_type: np.dtype) -> np.dtype:
        """
        Return the smallest dtype which can represent all elements of self.
        """
        if self.is_all_null:
            return self.dtype

        min_value, max_value = self.minmax()
        either_is_inf = np.isinf(min_value) or np.isinf(max_value)
        if not either_is_inf and expected_type.kind == "i":
            max_bound_dtype = min_signed_type(max_value)
            min_bound_dtype = min_signed_type(min_value)
            return np.promote_types(max_bound_dtype, min_bound_dtype)
        elif not either_is_inf and expected_type.kind == "u":
            max_bound_dtype = min_unsigned_type(max_value)
            min_bound_dtype = min_unsigned_type(min_value)
            return np.promote_types(max_bound_dtype, min_bound_dtype)
        elif self.dtype.kind == "f" or expected_type.kind == "f":
            return np.promote_types(
                expected_type,
                np.promote_types(
                    np.min_scalar_type(float(max_value)),
                    np.min_scalar_type(float(min_value)),
                ),
            )
        else:
            return self.dtype

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """
        Return col with *to_replace* replaced with *value*.
        """
        # TODO: all_nan and list arguments only used for this
        # this subclass, try to factor these cases out of this method

        # If all of `to_replace`/`replacement` are `None`,
        # dtype of `to_replace_col`/`replacement_col`
        # is inferred as `string`, but this is a valid
        # float64 column too, Hence we will need to type-cast
        # to self.dtype.
        to_replace_col = as_column(to_replace)
        if to_replace_col.is_all_null:
            to_replace_col = to_replace_col.astype(self.dtype)

        replacement_col = as_column(replacement)
        if replacement_col.is_all_null:
            replacement_col = replacement_col.astype(self.dtype)

        if not isinstance(to_replace_col, type(replacement_col)):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )

        if not isinstance(to_replace_col, NumericalColumn) and not isinstance(
            replacement_col, NumericalColumn
        ):
            return self.copy()

        try:
            to_replace_col = _normalize_find_and_replace_input(
                self.dtype, to_replace
            )
        except TypeError:
            # if `to_replace` cannot be normalized to the current dtype,
            # that means no value of `to_replace` is present in self,
            # Hence there is no point of proceeding further.
            return self.copy()

        if all_nan:
            replacement_col = as_column(replacement, dtype=self.dtype)
        else:
            try:
                replacement_col = _normalize_find_and_replace_input(
                    self.dtype, replacement
                )
            except TypeError:
                # Some floating values can never be converted into signed or unsigned integers
                # for those cases, we just need a column of `replacement` constructed
                # with its own type for the final type determination below at `find_common_type`
                # call.
                replacement_col = as_column(
                    replacement,
                    dtype=self.dtype if len(replacement) <= 0 else None,
                )
        common_type = find_common_type(
            (to_replace_col.dtype, replacement_col.dtype, self.dtype)
        )
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = replacement_col.repeat(len(to_replace_col))
        elif len(replacement_col) == 1 and len(to_replace_col) == 0:
            return self.copy()
        replaced = cast(Self, self.astype(common_type))
        df = cudf.DataFrame._from_data(
            {
                "old": to_replace_col.astype(common_type),
                "new": replacement_col.astype(common_type),
            }
        )
        df = df.drop_duplicates(subset=["old"], keep="last", ignore_index=True)
        if df._data["old"].null_count == 1:
            replaced = replaced.fillna(
                df._data["new"]
                .apply_boolean_mask(df._data["old"].isnull())
                .element_indexing(0)
            )
            df = df.dropna(subset=["old"])

        return replaced.replace(df._data["old"], df._data["new"])

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            cudf_obj = ColumnBase.from_pylibcudf(
                plc.Column.from_scalar(
                    pa_scalar_to_plc_scalar(pa.scalar(fill_value)), 1
                )
            )
            if not cudf_obj.can_cast_safely(self.dtype):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{type(fill_value).__name__} to {self.dtype.name}"
                )
            return super()._validate_fillna_value(fill_value)
        else:
            cudf_obj = as_column(fill_value, nan_as_null=False)
            if not cudf_obj.can_cast_safely(self.dtype):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{cudf_obj.dtype.type.__name__} to "
                    f"{self.dtype.type.__name__}"
                )
            return cudf_obj.astype(self.dtype)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        """
        Returns true if all the values in self can be
        safely cast to dtype
        """
        # Convert potential pandas extension dtypes to numpy dtypes
        # For example, convert Int32Dtype to np.dtype('int32')
        self_dtype_numpy = (
            np.dtype(self.dtype.numpy_dtype)
            if hasattr(self.dtype, "numpy_dtype")
            else self.dtype
        )
        to_dtype_numpy = (
            np.dtype(to_dtype.numpy_dtype)
            if hasattr(to_dtype, "numpy_dtype")
            else to_dtype
        )

        if self_dtype_numpy.kind == to_dtype_numpy.kind:
            # Check if self dtype can be safely cast to to_dtype
            # For same kinds, we can compare the sizes
            if self_dtype_numpy <= to_dtype_numpy:
                return True
            else:
                if self_dtype_numpy.kind == "f":
                    # Exclude 'np.inf', '-np.inf'
                    not_inf = (self != np.inf) & (self != -np.inf)
                    col = self.apply_boolean_mask(not_inf)
                else:
                    col = self

                min_ = col.min()
                # TODO: depending on implementation of cudf scalar and future
                # refactor of min/max, change the test method
                if np.isnan(min_):
                    # Column contains only infs
                    return True

                # Kinds are the same but to_dtype is smaller
                if "float" in to_dtype_numpy.name:
                    finfo = np.finfo(to_dtype_numpy)
                    lower_: int | float
                    upper_: int | float
                    # assignment ignore not needed for numpy>=2.4.0
                    lower_, upper_ = finfo.min, finfo.max  # type: ignore[assignment,unused-ignore]

                    # Check specifically for np.pi values when casting to lower precision
                    if self_dtype_numpy.itemsize > to_dtype_numpy.itemsize:
                        # Check if column contains pi value
                        if len(col) > 0:
                            # Create a simple column with pi to test if the precision matters
                            pi_col = self == np.pi
                            # Test if pi can be correctly represented after casting
                            if pi_col.any():
                                # If pi is present, we cannot safely cast to lower precision
                                return False
                elif "int" in to_dtype_numpy.name:
                    iinfo = np.iinfo(to_dtype_numpy)
                    lower_, upper_ = iinfo.min, iinfo.max

                return (min_ >= lower_) and (col.max() < upper_)

        # want to cast int to uint
        elif self_dtype_numpy.kind == "i" and to_dtype_numpy.kind == "u":
            i_max_ = np.iinfo(self_dtype_numpy).max
            u_max_ = np.iinfo(to_dtype_numpy).max

            return (self.min() >= 0) and (
                (i_max_ <= u_max_) or (self.max() < u_max_)
            )

        # want to cast uint to int
        elif self_dtype_numpy.kind == "u" and to_dtype_numpy.kind == "i":
            u_max_ = np.iinfo(self_dtype_numpy).max
            i_max_ = np.iinfo(to_dtype_numpy).max

            return (u_max_ <= i_max_) or (self.max() < i_max_)

        # want to cast int to float
        elif (
            self_dtype_numpy.kind in {"i", "u"} and to_dtype_numpy.kind == "f"
        ):
            info = np.finfo(to_dtype_numpy)
            biggest_exact_int = 2 ** (info.nmant + 1)
            if (self.min() >= -biggest_exact_int) and (
                self.max() <= biggest_exact_int
            ):
                return True
            else:
                filled = self.fillna(0)
                return (
                    filled.astype(to_dtype).astype(filled.dtype) == filled
                ).all()

        # want to cast float to int:
        elif self_dtype_numpy.kind == "f" and to_dtype_numpy.kind in {
            "i",
            "u",
        }:
            if self.nan_count > 0:
                return False
            iinfo = np.iinfo(to_dtype_numpy)
            min_, max_ = iinfo.min, iinfo.max

            # best we can do is hope to catch it here and avoid compare
            # Use Python floats, which have precise comparison for float64.
            # NOTE(seberg): it would make sense to limit to the mantissa range.
            min_val, max_val = self.minmax()
            if (float(min_val) >= min_) and (float(max_val) <= max_):
                filled = self.fillna(0)
                return (filled % 1 == 0).all()
            else:
                return False

        return False

    def _with_type_metadata(
        self: Self,
        dtype: DtypeObj,
    ) -> ColumnBase:
        if isinstance(dtype, CategoricalDtype):
            codes_dtype = min_unsigned_type(len(dtype.categories))
            # TODO: Try to avoid going via ColumnBase methods here
            codes = cast(
                cudf.core.column.numerical.NumericalColumn,
                self.astype(codes_dtype),
            )
            return CategoricalColumn._from_preprocessed(
                codes.plc_column, dtype
            )
        if cudf.get_option("mode.pandas_compatible"):
            res_dtype = get_dtype_of_same_type(dtype, self.dtype)
            if (
                is_pandas_nullable_extension_dtype(res_dtype)
                and isinstance(self.dtype, np.dtype)
                and self.dtype.kind == "f"
            ):
                # If the dtype is a pandas nullable extension type, we need to
                # float column doesn't have any NaNs.
                res = self.nans_to_nulls()
                res._dtype = res_dtype
                return res
            self._dtype = res_dtype

        return self

    def _reduction_result_dtype(self, reduction_op: str) -> DtypeObj:
        if reduction_op in {"sum", "product"}:
            if self.dtype.kind == "f":
                return self.dtype
            elif self.dtype.kind == "u":
                return np.dtype("uint64")
            return np.dtype("int64")
        elif reduction_op == "sum_of_squares":
            return find_common_type((self.dtype, np.dtype(np.uint64)))
        elif reduction_op in {"var", "std", "mean"}:
            if self.dtype.kind == "f":
                return self.dtype
            else:
                return np.dtype("float64")

        return super()._reduction_result_dtype(reduction_op)

    def digitize(self, bins: np.ndarray, right: bool = False) -> Self:
        """Return the indices of the bins to which each value in column belongs.

        Parameters
        ----------
        bins : np.ndarray
            1-D column-like object of bins with same type as `column`, should be
            monotonically increasing.
        right : bool
            Indicates whether interval contains the right or left bin edge.

        Returns
        -------
        A column containing the indices
        """
        if self.dtype != bins.dtype:
            raise ValueError(
                "digitize() expects bins and input column have the same dtype."
            )

        bin_col = as_column(bins, dtype=bins.dtype)
        if bin_col.nullable:
            raise ValueError("`bins` cannot contain null entries.")

        with access_columns(bin_col, self, mode="read", scope="internal") as (
            bin_col,
            self,
        ):
            return cast(
                Self,
                ColumnBase.create(
                    getattr(
                        plc.search, "lower_bound" if right else "upper_bound"
                    )(
                        plc.Table([bin_col.plc_column]),
                        plc.Table([self.plc_column]),
                        [plc.types.Order.ASCENDING],
                        [plc.types.NullOrder.BEFORE],
                    ),
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )


def _normalize_find_and_replace_input(
    input_column_dtype: DtypeObj, col_to_normalize: ColumnBase | list
) -> ColumnBase:
    normalized_column = as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        if normalized_column.is_all_null:
            normalized_column = normalized_column.astype(input_column_dtype)
        if normalized_column.can_cast_safely(input_column_dtype):
            return normalized_column.astype(input_column_dtype)
        col_to_normalize_dtype = normalized_column._min_column_type(  # type: ignore[attr-defined]
            input_column_dtype
        )
        # Scalar case
        if len(col_to_normalize) == 1:
            if is_na_like(col_to_normalize[0]):
                return normalized_column.astype(input_column_dtype)
            if np.isinf(col_to_normalize[0]):
                return normalized_column
            col_to_normalize_casted = np.array(col_to_normalize[0]).astype(
                col_to_normalize_dtype
            )

            if not np.isnan(col_to_normalize_casted) and (
                col_to_normalize_casted != col_to_normalize[0]
            ):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{col_to_normalize[0]} "
                    f"to {input_column_dtype.name}"
                )
        if normalized_column.can_cast_safely(col_to_normalize_dtype):
            return normalized_column.astype(col_to_normalize_dtype)
    elif hasattr(col_to_normalize, "dtype"):
        col_to_normalize_dtype = col_to_normalize.dtype
    else:
        raise TypeError(f"Type {type(col_to_normalize)} not supported")

    if (
        col_to_normalize_dtype.kind == "f"
        and input_column_dtype.kind in {"i", "u"}
    ) or (col_to_normalize_dtype.num > input_column_dtype.num):
        raise TypeError(
            f"Potentially unsafe cast for non-equivalent "
            f"{col_to_normalize_dtype.name} "
            f"to {input_column_dtype.name}"
        )
    if not normalized_column.can_cast_safely(input_column_dtype):
        return normalized_column
    return normalized_column.astype(input_column_dtype)
