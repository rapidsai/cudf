# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from typing_extensions import Self

import pylibcudf

import cudf
from cudf import _lib as libcudf
from cudf.api.types import is_integer, is_scalar
from cudf.core._internals import unary
from cudf.core.column import ColumnBase, as_column, column, string
from cudf.core.dtypes import CategoricalDtype
from cudf.core.mixins import BinaryOperand
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    find_common_type,
    min_column_type,
    min_signed_type,
    np_dtypes_to_pandas_dtypes,
)

from .numerical_base import NumericalBaseColumn

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        Dtype,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.buffer import Buffer

_unaryop_map = {
    "ASIN": "ARCSIN",
    "ACOS": "ARCCOS",
    "ATAN": "ARCTAN",
    "INVERT": "BIT_INVERT",
}


class NumericalColumn(NumericalBaseColumn):
    """
    A Column object for Numeric types.

    Parameters
    ----------
    data : Buffer
    dtype : np.dtype
        The dtype associated with the data Buffer
    mask : Buffer, optional
    """

    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(
        self,
        data: Buffer,
        size: int | None,
        dtype: np.dtype,
        mask: Buffer | None = None,
        offset: int = 0,
        null_count: int | None = None,
        children: tuple = (),
    ):
        if not (isinstance(dtype, np.dtype) and dtype.kind in "iufb"):
            raise ValueError(
                "dtype must be a floating, integer or boolean numpy dtype."
            )

        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = (data.size // dtype.itemsize) - offset
        super().__init__(
            data=data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

    def _clear_cache(self):
        super()._clear_cache()
        try:
            del self.nan_count
        except AttributeError:
            pass

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
        return libcudf.search.contains(
            self, column.as_column([search_item], dtype=self.dtype)
        ).any()

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        if isinstance(value, (bool, np.bool_)) and self.dtype.kind != "b":
            raise ValueError(
                f"Cannot use a {type(value).__name__} to find an index of "
                f"a {self.dtype} Index."
            )
        if (
            value is not None
            and self.dtype.kind in {"c", "f"}
            and np.isnan(value)
        ):
            nan_col = unary.is_nan(self)
            return nan_col.indices_of(True)
        else:
            return super().indices_of(value)

    def has_nulls(self, include_nan: bool = False) -> bool:
        return bool(self.null_count != 0) or (
            include_nan and bool(self.nan_count != 0)
        )

    def __setitem__(self, key: Any, value: Any):
        """
        Set the value of ``self[key]`` to ``value``.

        If ``value`` and ``self`` are of different types, ``value`` is coerced
        to ``self.dtype``.
        """

        # Normalize value to scalar/column
        device_value: cudf.Scalar | ColumnBase = (
            cudf.Scalar(
                value,
                dtype=self.dtype
                if cudf._lib.scalar._is_null_host_scalar(value)
                else None,
            )
            if is_scalar(value)
            else as_column(value)
        )

        if self.dtype.kind != "b" and device_value.dtype.kind == "b":
            raise TypeError(f"Invalid value {value} for dtype {self.dtype}")
        else:
            device_value = device_value.astype(self.dtype)

        out: ColumnBase | None  # If None, no need to perform mimic inplace.
        if isinstance(key, slice):
            out = self._scatter_by_slice(key, device_value)
        else:
            key = as_column(
                key,
                dtype="float64"
                if isinstance(key, list) and len(key) == 0
                else None,
            )
            if not isinstance(key, cudf.core.column.NumericalColumn):
                raise ValueError(f"Invalid scatter map type {key.dtype}.")
            out = self._scatter_by_column(key, device_value)

        if out:
            self._mimic_inplace(out, inplace=True)

    def unary_operator(self, unaryop: str | Callable) -> ColumnBase:
        if callable(unaryop):
            return libcudf.transform.transform(self, unaryop)

        unaryop = unaryop.upper()
        unaryop = _unaryop_map.get(unaryop, unaryop)
        unaryop = pylibcudf.unary.UnaryOperator[unaryop]
        return unary.unary_operation(self, unaryop)

    def __invert__(self):
        if self.dtype.kind in "ui":
            return self.unary_operator("invert")
        elif self.dtype.kind == "b":
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

        out_dtype = None
        if op in {"__truediv__", "__rtruediv__"}:
            # Division with integer types results in a suitable float.
            if truediv_type := int_float_dtype_mapping.get(self.dtype.type):
                return self.astype(truediv_type)._binaryop(other, op)
        elif op in {
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
            "__eq__",
            "__ne__",
        }:
            out_dtype = "bool"

            # If `other` is a Python integer and it is out-of-bounds
            # promotion could fail but we can trivially define the result
            # in terms of `notnull` or `NULL_NOT_EQUALS`.
            if type(other) is int and self.dtype.kind in "iu":  # noqa: E721
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
            out_dtype = "bool"

        reflect, op = self._check_reflected_op(op)
        if (other := self._wrap_binop_normalization(other)) is NotImplemented:
            return NotImplemented

        if out_dtype is not None:
            pass  # out_dtype was already set to bool
        if other is None:
            # not a binary operator, so no need to promote
            out_dtype = self.dtype
        elif out_dtype is None:
            out_dtype = np.result_type(self.dtype, other.dtype)
            if op in {"__mod__", "__floordiv__"}:
                tmp = self if reflect else other
                # Guard against division by zero for integers.
                if (
                    tmp.dtype.type in int_float_dtype_mapping
                    and tmp.dtype.kind != "b"
                ):
                    if isinstance(tmp, NumericalColumn) and 0 in tmp:
                        out_dtype = cudf.dtype("float64")
                    elif isinstance(tmp, cudf.Scalar):
                        if tmp.is_valid() and tmp == 0:
                            # tmp == 0 can return NA
                            out_dtype = cudf.dtype("float64")
                    elif is_scalar(tmp) and tmp == 0:
                        out_dtype = cudf.dtype("float64")

        if op in {"__and__", "__or__", "__xor__"}:
            if self.dtype.kind == "f" or other.dtype.kind == "f":
                raise TypeError(
                    f"Operation 'bitwise {op[2:-2]}' not supported between "
                    f"{self.dtype.type.__name__} and "
                    f"{other.dtype.type.__name__}"
                )
            if self.dtype.kind == "b" or other.dtype.kind == "b":
                out_dtype = "bool"

        elif (
            op == "__pow__"
            and self.dtype.kind in "iu"
            and (is_integer(other) or other.dtype.kind in "iu")
        ):
            op = "INT_POW"

        lhs, rhs = (other, self) if reflect else (self, other)

        return libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)

    def nans_to_nulls(self: Self) -> Self:
        # Only floats can contain nan.
        if self.dtype.kind != "f" or self.nan_count == 0:
            return self
        newmask = libcudf.transform.nans_to_nulls(self)
        return self.set_mask(newmask)

    def normalize_binop_value(
        self, other: ScalarLike
    ) -> ColumnBase | cudf.Scalar:
        if isinstance(other, ColumnBase):
            if not isinstance(other, NumericalColumn):
                return NotImplemented
            return other
        if isinstance(other, cudf.Scalar):
            if self.dtype == other.dtype:
                return other

            # expensive device-host transfer just to
            # adjust the dtype
            other = other.value

            # NumPy 2 needs a Python scalar to do weak promotion, but
            # pandas forces weak promotion always
            # TODO: We could use 0, 0.0, and 0j for promotion to avoid copies.
            if other.dtype.kind in "ifc":
                other = other.item()
        elif not isinstance(other, (int, float, complex)):
            # Go via NumPy to get the value
            other = np.array(other)
            if other.dtype.kind in "ifc":
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
        common_dtype = np.result_type(self.dtype, other)
        if common_dtype.kind in {"b", "i", "u", "f"}:
            if self.dtype.kind == "b":
                common_dtype = min_signed_type(other)
            return cudf.Scalar(other, dtype=common_dtype)
        else:
            return NotImplemented

    def int2ip(self) -> "cudf.core.column.StringColumn":
        if self.dtype != cudf.dtype("uint32"):
            raise TypeError("Only uint32 type can be converted to ip")

        return libcudf.string_casting.int2ip(self)

    def as_string_column(self) -> cudf.core.column.StringColumn:
        if len(self) > 0:
            return string._numeric_to_str_typecast_functions[
                cudf.dtype(self.dtype)
            ](self)
        else:
            return cast(
                cudf.core.column.StringColumn,
                column.column_empty(0, dtype="object"),
            )

    def as_datetime_column(
        self, dtype: Dtype
    ) -> cudf.core.column.DatetimeColumn:
        return cudf.core.column.DatetimeColumn(
            data=self.astype("int64").base_data,  # type: ignore[arg-type]
            dtype=dtype,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )

    def as_timedelta_column(
        self, dtype: Dtype
    ) -> cudf.core.column.TimeDeltaColumn:
        return cudf.core.column.TimeDeltaColumn(
            data=self.astype("int64").base_data,  # type: ignore[arg-type]
            dtype=dtype,
            mask=self.base_mask,
            offset=self.offset,
            size=self.size,
        )

    def as_decimal_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.DecimalBaseColumn":
        return unary.cast(self, dtype)  # type: ignore[return-value]

    def as_numerical_column(self, dtype: Dtype) -> NumericalColumn:
        dtype = cudf.dtype(dtype)
        if dtype == self.dtype:
            return self
        return unary.cast(self, dtype)  # type: ignore[return-value]

    def all(self, skipna: bool = True) -> bool:
        # If all entries are null the result is True, including when the column
        # is empty.
        result_col = self.nans_to_nulls() if skipna else self

        if result_col.null_count == result_col.size:
            return True

        return libcudf.reduce.reduce("all", result_col)

    def any(self, skipna: bool = True) -> bool:
        # Early exit for fast cases.
        result_col = self.nans_to_nulls() if skipna else self

        if not skipna and result_col.has_nulls():
            return True
        elif skipna and result_col.null_count == result_col.size:
            return False

        return libcudf.reduce.reduce("any", result_col)

    @functools.cached_property
    def nan_count(self) -> int:
        if self.dtype.kind != "f":
            return 0
        nan_col = unary.is_nan(self)
        return nan_col.sum()

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
        lhs = cast("cudf.core.column.ColumnBase", self)
        try:
            rhs = as_column(values, nan_as_null=False)
        except (MixedTypeError, TypeError) as e:
            # There is a corner where `values` can be of `object` dtype
            # but have values of homogeneous type.
            inferred_dtype = cudf.api.types.infer_dtype(values)
            if (
                self.dtype.kind in {"i", "u"} and inferred_dtype == "integer"
            ) or (
                self.dtype.kind == "f"
                and inferred_dtype in {"floating", "integer"}
            ):
                rhs = as_column(values, nan_as_null=False, dtype=self.dtype)
            elif self.dtype.kind == "f" and inferred_dtype == "integer":
                rhs = as_column(values, nan_as_null=False, dtype="int")
            elif (
                self.dtype.kind in {"i", "u"} and inferred_dtype == "floating"
            ):
                rhs = as_column(values, nan_as_null=False, dtype="float")
            else:
                raise e
        else:
            if isinstance(rhs, NumericalColumn):
                rhs = rhs.astype(dtype=self.dtype)

        if lhs.null_count == len(lhs):
            lhs = lhs.astype(rhs.dtype)
        elif rhs.null_count == len(rhs):
            rhs = rhs.astype(lhs.dtype)

        return lhs, rhs

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls(include_nan=True)

    def _process_for_reduction(
        self, skipna: bool | None = None, min_count: int = 0
    ) -> NumericalColumn | ScalarLike:
        skipna = True if skipna is None else skipna

        if self._can_return_nan(skipna=skipna):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        col = self.nans_to_nulls() if skipna else self
        return super(NumericalColumn, col)._process_for_reduction(
            skipna=skipna, min_count=min_count
        )

    def find_and_replace(
        self,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> NumericalColumn:
        """
        Return col with *to_replace* replaced with *value*.
        """

        # If all of `to_replace`/`replacement` are `None`,
        # dtype of `to_replace_col`/`replacement_col`
        # is inferred as `string`, but this is a valid
        # float64 column too, Hence we will need to type-cast
        # to self.dtype.
        to_replace_col = column.as_column(to_replace)
        if to_replace_col.null_count == len(to_replace_col):
            to_replace_col = to_replace_col.astype(self.dtype)

        replacement_col = column.as_column(replacement)
        if replacement_col.null_count == len(replacement_col):
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
            replacement_col = column.as_column(replacement, dtype=self.dtype)
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
                replacement_col = column.as_column(
                    replacement,
                    dtype=self.dtype if len(replacement) <= 0 else None,
                )
        common_type = find_common_type(
            (to_replace_col.dtype, replacement_col.dtype, self.dtype)
        )
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = column.as_column(
                replacement[0], length=len(to_replace_col), dtype=common_type
            )
        elif len(replacement_col) == 1 and len(to_replace_col) == 0:
            return self.copy()
        replaced = self.astype(common_type)
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

        return libcudf.replace.replace(
            replaced, df._data["old"], df._data["new"]
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> cudf.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            cudf_obj: cudf.Scalar | ColumnBase = cudf.Scalar(fill_value)
            if not as_column(cudf_obj).can_cast_safely(self.dtype):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{type(fill_value).__name__} to {self.dtype.name}"
                )
        else:
            cudf_obj = as_column(fill_value, nan_as_null=False)
            if not cudf_obj.can_cast_safely(self.dtype):  # type: ignore[attr-defined]
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
        if self.dtype.kind == to_dtype.kind:
            if self.dtype <= to_dtype:
                return True
            else:
                # Kinds are the same but to_dtype is smaller
                if "float" in to_dtype.name:
                    finfo = np.finfo(to_dtype)
                    lower_, upper_ = finfo.min, finfo.max
                elif "int" in to_dtype.name:
                    iinfo = np.iinfo(to_dtype)
                    lower_, upper_ = iinfo.min, iinfo.max

                if self.dtype.kind == "f":
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

                return (min_ >= lower_) and (col.max() < upper_)

        # want to cast int to uint
        elif self.dtype.kind == "i" and to_dtype.kind == "u":
            i_max_ = np.iinfo(self.dtype).max
            u_max_ = np.iinfo(to_dtype).max

            return (self.min() >= 0) and (
                (i_max_ <= u_max_) or (self.max() < u_max_)
            )

        # want to cast uint to int
        elif self.dtype.kind == "u" and to_dtype.kind == "i":
            u_max_ = np.iinfo(self.dtype).max
            i_max_ = np.iinfo(to_dtype).max

            return (u_max_ <= i_max_) or (self.max() < i_max_)

        # want to cast int to float
        elif self.dtype.kind in {"i", "u"} and to_dtype.kind == "f":
            info = np.finfo(to_dtype)
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
        elif self.dtype.kind == "f" and to_dtype.kind in {"i", "u"}:
            if self.nan_count > 0:
                return False
            iinfo = np.iinfo(to_dtype)
            min_, max_ = iinfo.min, iinfo.max

            # best we can do is hope to catch it here and avoid compare
            # Use Python floats, which have precise comparison for float64.
            # NOTE(seberg): it would make sense to limit to the mantissa range.
            if (float(self.min()) >= min_) and (float(self.max()) <= max_):
                filled = self.fillna(0)
                return (filled % 1 == 0).all()
            else:
                return False

        return False

    def _with_type_metadata(self: Self, dtype: Dtype) -> ColumnBase:
        if isinstance(dtype, CategoricalDtype):
            codes = cudf.core.column.categorical.as_unsigned_codes(
                len(dtype.categories), self
            )
            return cudf.core.column.CategoricalColumn(
                data=None,
                size=self.size,
                dtype=dtype,
                mask=self.base_mask,
                offset=self.offset,
                null_count=self.null_count,
                children=(codes,),
            )
        return self

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type and nullable:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif arrow_type:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif (
            nullable
            and (
                pandas_nullable_dtype := np_dtypes_to_pandas_dtypes.get(
                    self.dtype
                )
            )
            is not None
        ):
            arrow_array = self.to_arrow()
            pandas_array = pandas_nullable_dtype.__from_arrow__(arrow_array)  # type: ignore[attr-defined]
            return pd.Index(pandas_array, copy=False)
        elif self.dtype.kind in set("iuf") and not self.has_nulls():
            return pd.Index(self.values_host, copy=False)
        else:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)

    def _reduction_result_dtype(self, reduction_op: str) -> Dtype:
        if reduction_op in {"sum", "product"}:
            if self.dtype.kind == "f":
                return self.dtype
            return np.dtype("int64")
        elif reduction_op == "sum_of_squares":
            return np.result_dtype(self.dtype, np.dtype("uint64"))
        elif reduction_op in {"var", "std", "mean"}:
            return np.dtype("float64")

        return super()._reduction_result_dtype(reduction_op)


def _normalize_find_and_replace_input(
    input_column_dtype: DtypeObj, col_to_normalize: ColumnBase | list
) -> ColumnBase:
    normalized_column = column.as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        if normalized_column.null_count == len(normalized_column):
            normalized_column = normalized_column.astype(input_column_dtype)
        if normalized_column.can_cast_safely(input_column_dtype):
            return normalized_column.astype(input_column_dtype)
        col_to_normalize_dtype = min_column_type(
            normalized_column, input_column_dtype
        )
        # Scalar case
        if len(col_to_normalize) == 1:
            if cudf._lib.scalar._is_null_host_scalar(col_to_normalize[0]):
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


def digitize(
    column: ColumnBase, bins: np.ndarray, right: bool = False
) -> ColumnBase:
    """Return the indices of the bins to which each value in column belongs.

    Parameters
    ----------
    column : Column
        Input column.
    bins : Column-like
        1-D column-like object of bins with same type as `column`, should be
        monotonically increasing.
    right : bool
        Indicates whether interval contains the right or left bin edge.

    Returns
    -------
    A column containing the indices
    """
    if not column.dtype == bins.dtype:
        raise ValueError(
            "Digitize() expects bins and input column have the same dtype."
        )

    bin_col = as_column(bins, dtype=bins.dtype)
    if bin_col.nullable:
        raise ValueError("`bins` cannot contain null entries.")

    return as_column(libcudf.sort.digitize([column], [bin_col], right))
