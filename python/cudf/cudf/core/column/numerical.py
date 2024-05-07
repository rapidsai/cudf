# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union, cast

import cupy as cp
import numpy as np
import pandas as pd
from typing_extensions import Self

import cudf
from cudf import _lib as libcudf
from cudf._lib import pylibcudf
from cudf._lib.types import size_type_dtype
from cudf._typing import (
    ColumnBinaryOperand,
    ColumnLike,
    Dtype,
    DtypeObj,
    ScalarLike,
)
from cudf.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_scalar,
)
from cudf.core.buffer import Buffer
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_column,
    column,
    string,
)
from cudf.core.dtypes import CategoricalDtype
from cudf.core.mixins import BinaryOperand
from cudf.utils.dtypes import (
    min_column_type,
    min_signed_type,
    np_dtypes_to_pandas_dtypes,
    numeric_normalize_types,
)

from .numerical_base import NumericalBaseColumn

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
        dtype: DtypeObj,
        mask: Optional[Buffer] = None,
        size: Optional[int] = None,  # TODO: make this non-optional
        offset: int = 0,
        null_count: Optional[int] = None,
    ):
        dtype = cudf.dtype(dtype)

        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = (data.size // dtype.itemsize) - offset
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
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
            return column.as_column(
                cp.argwhere(
                    cp.isnan(self.data_array_view(mode="read"))
                ).flatten(),
                dtype=size_type_dtype,
            )
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
        device_value = (
            cudf.Scalar(
                value,
                dtype=self.dtype
                if cudf._lib.scalar._is_null_host_scalar(value)
                else None,
            )
            if is_scalar(value)
            else as_column(value)
        )

        if not is_bool_dtype(self.dtype) and is_bool_dtype(device_value.dtype):
            raise TypeError(f"Invalid value {value} for dtype {self.dtype}")
        else:
            device_value = device_value.astype(self.dtype)

        out: Optional[ColumnBase]  # If None, no need to perform mimic inplace.
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

    def unary_operator(self, unaryop: Union[str, Callable]) -> ColumnBase:
        if callable(unaryop):
            return libcudf.transform.transform(self, unaryop)

        unaryop = unaryop.upper()
        unaryop = _unaryop_map.get(unaryop, unaryop)
        unaryop = pylibcudf.unary.UnaryOperator[unaryop]
        return libcudf.unary.unary_operation(self, unaryop)

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

        if op in {"__truediv__", "__rtruediv__"}:
            # Division with integer types results in a suitable float.
            if truediv_type := int_float_dtype_mapping.get(self.dtype.type):
                return self.astype(truediv_type)._binaryop(other, op)

        reflect, op = self._check_reflected_op(op)
        if (other := self._wrap_binop_normalization(other)) is NotImplemented:
            return NotImplemented
        out_dtype = self.dtype
        if other is not None:
            out_dtype = np.result_type(self.dtype, other.dtype)
            if op in {"__mod__", "__floordiv__"}:
                tmp = self if reflect else other
                # Guard against division by zero for integers.
                if (
                    (tmp.dtype.type in int_float_dtype_mapping)
                    and (tmp.dtype.type != np.bool_)
                    and (
                        (
                            (
                                np.isscalar(tmp)
                                or (
                                    isinstance(tmp, cudf.Scalar)
                                    # host to device copy
                                    and tmp.is_valid()
                                )
                            )
                            and (0 == tmp)
                        )
                        or ((isinstance(tmp, NumericalColumn)) and (0 in tmp))
                    )
                ):
                    out_dtype = cudf.dtype("float64")

        if op in {
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
            "__eq__",
            "__ne__",
            "NULL_EQUALS",
        }:
            out_dtype = "bool"

        if op in {"__and__", "__or__", "__xor__"}:
            if is_float_dtype(self.dtype) or is_float_dtype(other.dtype):
                raise TypeError(
                    f"Operation 'bitwise {op[2:-2]}' not supported between "
                    f"{self.dtype.type.__name__} and "
                    f"{other.dtype.type.__name__}"
                )
            if is_bool_dtype(self.dtype) or is_bool_dtype(other.dtype):
                out_dtype = "bool"

        if (
            op == "__pow__"
            and is_integer_dtype(self.dtype)
            and (is_integer(other) or is_integer_dtype(other.dtype))
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
    ) -> Union[ColumnBase, cudf.Scalar]:
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
        # Try and match pandas and hence numpy. Deduce the common
        # dtype via the _value_ of other, and the dtype of self. TODO:
        # When NEP50 is accepted, this might want changed or
        # simplified.
        # This is not at all simple:
        # np.result_type(np.int64(0), np.uint8)
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
        if self.dtype != cudf.dtype("int64"):
            raise TypeError("Only int64 type can be converted to ip")

        return libcudf.string_casting.int2ip(self)

    def as_string_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.StringColumn":
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
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.DatetimeColumn":
        return cast(
            "cudf.core.column.DatetimeColumn",
            build_column(
                data=self.astype("int64").base_data,
                dtype=dtype,
                mask=self.base_mask,
                offset=self.offset,
                size=self.size,
            ),
        )

    def as_timedelta_column(
        self, dtype: Dtype, format: str | None = None
    ) -> "cudf.core.column.TimeDeltaColumn":
        return cast(
            "cudf.core.column.TimeDeltaColumn",
            build_column(
                data=self.astype("int64").base_data,
                dtype=dtype,
                mask=self.base_mask,
                offset=self.offset,
                size=self.size,
            ),
        )

    def as_decimal_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.DecimalBaseColumn":
        return libcudf.unary.cast(self, dtype)

    def as_numerical_column(self, dtype: Dtype) -> NumericalColumn:
        dtype = cudf.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype)

    def all(self, skipna: bool = True) -> bool:
        # If all entries are null the result is True, including when the column
        # is empty.
        result_col = self.nans_to_nulls() if skipna else self

        if result_col.null_count == result_col.size:
            return True

        return libcudf.reduce.reduce("all", result_col, dtype=np.bool_)

    def any(self, skipna: bool = True) -> bool:
        # Early exit for fast cases.
        result_col = self.nans_to_nulls() if skipna else self

        if not skipna and result_col.has_nulls():
            return True
        elif skipna and result_col.null_count == result_col.size:
            return False

        return libcudf.reduce.reduce("any", result_col, dtype=np.bool_)

    @functools.cached_property
    def nan_count(self) -> int:
        if self.dtype.kind != "f":
            return 0
        nan_col = libcudf.unary.is_nan(self)
        return nan_col.sum()

    def _process_values_for_isin(
        self, values: Sequence
    ) -> Tuple[ColumnBase, ColumnBase]:
        lhs = cast("cudf.core.column.ColumnBase", self)
        rhs = as_column(values, nan_as_null=False)

        if isinstance(rhs, NumericalColumn):
            rhs = rhs.astype(dtype=self.dtype)

        if lhs.null_count == len(lhs):
            lhs = lhs.astype(rhs.dtype)
        elif rhs.null_count == len(rhs):
            rhs = rhs.astype(lhs.dtype)

        return lhs, rhs

    def _can_return_nan(self, skipna: Optional[bool] = None) -> bool:
        return not skipna and self.has_nulls(include_nan=True)

    def _process_for_reduction(
        self, skipna: Optional[bool] = None, min_count: int = 0
    ) -> Union[NumericalColumn, ScalarLike]:
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

        to_replace_col = _normalize_find_and_replace_input(
            self.dtype, to_replace
        )
        if all_nan:
            replacement_col = column.as_column(replacement, dtype=self.dtype)
        else:
            replacement_col = _normalize_find_and_replace_input(
                self.dtype, replacement
            )
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = column.as_column(
                replacement[0], length=len(to_replace_col), dtype=self.dtype
            )
        elif len(replacement_col) == 1 and len(to_replace_col) == 0:
            return self.copy()
        to_replace_col, replacement_col, replaced = numeric_normalize_types(
            to_replace_col, replacement_col, self
        )
        df = cudf.DataFrame._from_data(
            {"old": to_replace_col, "new": replacement_col}
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

    def fillna(
        self,
        fill_value: Any = None,
        method: Optional[str] = None,
    ) -> Self:
        """
        Fill null values with *fill_value*
        """
        col = self.nans_to_nulls()

        if col.null_count == 0:
            return col

        if method is not None:
            return super(NumericalColumn, col).fillna(fill_value, method)

        if fill_value is None:
            raise ValueError("Must specify either 'fill_value' or 'method'")

        if (
            isinstance(fill_value, cudf.Scalar)
            and fill_value.dtype == col.dtype
        ):
            return super(NumericalColumn, col).fillna(fill_value, method)

        if np.isscalar(fill_value):
            # cast safely to the same dtype as self
            fill_value_casted = col.dtype.type(fill_value)
            if not np.isnan(fill_value) and (fill_value_casted != fill_value):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{type(fill_value).__name__} to {col.dtype.name}"
                )
            fill_value = cudf.Scalar(fill_value_casted)
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
            if is_integer_dtype(col.dtype):
                # cast safely to the same dtype as self
                if fill_value.dtype != col.dtype:
                    new_fill_value = fill_value.astype(col.dtype)
                    if not (new_fill_value == fill_value).all():
                        raise TypeError(
                            f"Cannot safely cast non-equivalent "
                            f"{fill_value.dtype.type.__name__} to "
                            f"{col.dtype.type.__name__}"
                        )
                    fill_value = new_fill_value
            else:
                fill_value = fill_value.astype(col.dtype)

        return super(NumericalColumn, col).fillna(fill_value, method)

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
                    s = cudf.Series(self)
                    # TODO: replace np.inf with cudf scalar when
                    # https://github.com/rapidsai/cudf/pull/6297 merges
                    non_infs = s[~((s == np.inf) | (s == -np.inf))]
                    col = non_infs._column
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
                    cudf.Series(filled).astype(to_dtype).astype(filled.dtype)
                    == cudf.Series(filled)
                ).all()

        # want to cast float to int:
        elif self.dtype.kind == "f" and to_dtype.kind in {"i", "u"}:
            if self.nan_count > 0:
                return False
            iinfo = np.iinfo(to_dtype)
            min_, max_ = iinfo.min, iinfo.max

            # best we can do is hope to catch it here and avoid compare
            if (self.min() >= min_) and (self.max() <= max_):
                filled = self.fillna(0)
                return (cudf.Series(filled) % 1 == 0).all()
            else:
                return False

        return False

    def _with_type_metadata(self: ColumnBase, dtype: Dtype) -> ColumnBase:
        if isinstance(dtype, CategoricalDtype):
            return column.build_categorical_column(
                categories=dtype.categories._values,
                codes=build_column(self.base_data, dtype=self.dtype),
                mask=self.base_mask,
                ordered=dtype.ordered,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )

        return self

    def to_pandas(
        self,
        *,
        index: Optional[pd.Index] = None,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Series:
        if arrow_type and nullable:
            raise ValueError(
                f"{arrow_type=} and {nullable=} cannot both be set."
            )
        elif arrow_type:
            return pd.Series(
                pd.arrays.ArrowExtensionArray(self.to_arrow()), index=index
            )
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
            return pd.Series(pandas_array, copy=False, index=index)
        elif self.dtype.kind in set("iuf") and not self.has_nulls():
            return pd.Series(self.values_host, copy=False, index=index)
        else:
            return super().to_pandas(index=index, nullable=nullable)

    def _reduction_result_dtype(self, reduction_op: str) -> Dtype:
        col_dtype = self.dtype
        if reduction_op in {"sum", "product"}:
            col_dtype = (
                col_dtype if col_dtype.kind == "f" else np.dtype("int64")
            )
        elif reduction_op == "sum_of_squares":
            col_dtype = np.result_dtype(col_dtype, np.dtype("uint64"))

        return col_dtype


def _normalize_find_and_replace_input(
    input_column_dtype: DtypeObj, col_to_normalize: Union[ColumnBase, list]
) -> ColumnBase:
    normalized_column = column.as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        if normalized_column.null_count == len(normalized_column):
            normalized_column = normalized_column.astype(input_column_dtype)
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
                input_column_dtype
            )

            if not np.isnan(col_to_normalize_casted) and (
                col_to_normalize_casted != col_to_normalize[0]
            ):
                raise TypeError(
                    f"Cannot safely cast non-equivalent "
                    f"{col_to_normalize[0]} "
                    f"to {input_column_dtype.name}"
                )
            else:
                col_to_normalize_dtype = input_column_dtype
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
