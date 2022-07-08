# Copyright (c) 2018-2022, NVIDIA CORPORATION.

from __future__ import annotations

from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import cupy
import numpy as np
import pandas as pd

import cudf
from cudf import _lib as libcudf
from cudf._lib.stream_compaction import drop_nulls
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
    is_integer_dtype,
    is_number,
)
from cudf.core.buffer import Buffer
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_column,
    column,
    full,
    string,
)
from cudf.core.dtypes import CategoricalDtype
from cudf.core.mixins import BinaryOperand
from cudf.utils import cudautils
from cudf.utils.dtypes import (
    NUMERIC_TYPES,
    min_column_type,
    min_signed_type,
    np_dtypes_to_pandas_dtypes,
    numeric_normalize_types,
    to_cudf_compatible_scalar,
)

from .numerical_base import NumericalBaseColumn


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

    _nan_count: Optional[int]
    _VALID_BINARY_OPERATIONS = BinaryOperand._SUPPORTED_BINARY_OPERATIONS

    def __init__(
        self,
        data: Buffer,
        dtype: DtypeObj,
        mask: Buffer = None,
        size: int = None,  # TODO: make this non-optional
        offset: int = 0,
        null_count: int = None,
    ):
        dtype = cudf.dtype(dtype)

        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = (data.size // dtype.itemsize) - offset
        self._nan_count = None
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
        self._nan_count = None

    def __contains__(self, item: ScalarLike) -> bool:
        """
        Returns True if column contains item, else False.
        """
        # Handles improper item types
        # Fails if item is of type None, so the handler.
        try:
            if np.can_cast(item, self.data_array_view.dtype):
                item = self.data_array_view.dtype.type(item)
            else:
                return False
        except (TypeError, ValueError):
            return False
        # TODO: Use `scalar`-based `contains` wrapper
        return libcudf.search.contains(
            self, column.as_column([item], dtype=self.dtype)
        ).any()

    def has_nulls(self, include_nan=False):
        return self.null_count != 0 or (
            self.nan_count != 0 if include_nan else False
        )

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        output = {
            "shape": (len(self),),
            "strides": (self.dtype.itemsize,),
            "typestr": self.dtype.str,
            "data": (self.data_ptr, False),
            "version": 1,
        }

        if self.nullable and self.has_nulls():

            # Create a simple Python object that exposes the
            # `__cuda_array_interface__` attribute here since we need to modify
            # some of the attributes from the numba device array
            mask = SimpleNamespace(
                __cuda_array_interface__={
                    "shape": (len(self),),
                    "typestr": "<t1",
                    "data": (self.mask_ptr, True),
                    "version": 1,
                }
            )
            output["mask"] = mask

        return output

    def unary_operator(self, unaryop: Union[str, Callable]) -> ColumnBase:
        if callable(unaryop):
            return libcudf.transform.transform(self, unaryop)

        unaryop = libcudf.unary.UnaryOp[unaryop.upper()]
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
                        (np.isscalar(tmp) and (0 == tmp))
                        or (
                            (isinstance(tmp, NumericalColumn)) and (0.0 in tmp)
                        )
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
            if is_float_dtype(self.dtype) or is_float_dtype(other):
                raise TypeError(
                    f"Operation 'bitwise {op[2:-2]}' not supported between "
                    f"{self.dtype.type.__name__} and "
                    f"{other.dtype.type.__name__}"
                )
            if is_bool_dtype(self.dtype) or is_bool_dtype(other):
                out_dtype = "bool"

        lhs, rhs = (other, self) if reflect else (self, other)
        return libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)

    def nans_to_nulls(self: NumericalColumn) -> NumericalColumn:
        # Only floats can contain nan.
        if self.dtype.kind != "f" or self.nan_count == 0:
            return self
        newmask = libcudf.transform.nans_to_nulls(self)
        return self.set_mask(newmask)

    def normalize_binop_value(
        self, other: ScalarLike
    ) -> Union[ColumnBase, ScalarLike]:
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
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in {"b", "i", "u", "f"}:
            if isinstance(other, cudf.Scalar):
                return other
            other_dtype = np.promote_types(self.dtype, other_dtype)
            if other_dtype == np.dtype("float16"):
                other_dtype = cudf.dtype("float32")
                other = other_dtype.type(other)
            if self.dtype.kind == "b":
                other_dtype = min_signed_type(other)
            if np.isscalar(other):
                return cudf.dtype(other_dtype).type(other)
            else:
                ary = full(len(self), other, dtype=other_dtype)
                return column.build_column(
                    data=Buffer(ary),
                    dtype=ary.dtype,
                    mask=self.mask,
                )
        else:
            return NotImplemented

    def int2ip(self) -> "cudf.core.column.StringColumn":
        if self.dtype != cudf.dtype("int64"):
            raise TypeError("Only int64 type can be converted to ip")

        return libcudf.string_casting.int2ip(self)

    def as_string_column(
        self, dtype: Dtype, format=None, **kwargs
    ) -> "cudf.core.column.StringColumn":
        if len(self) > 0:
            return string._numeric_to_str_typecast_functions[
                cudf.dtype(self.dtype)
            ](self)
        else:
            return cast(
                "cudf.core.column.StringColumn", as_column([], dtype="object")
            )

    def as_datetime_column(
        self, dtype: Dtype, **kwargs
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
        self, dtype: Dtype, **kwargs
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
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.DecimalBaseColumn":
        return libcudf.unary.cast(self, dtype)

    def as_numerical_column(self, dtype: Dtype, **kwargs) -> NumericalColumn:
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

    @property
    def nan_count(self) -> int:
        if self.dtype.kind != "f":
            self._nan_count = 0
        elif self._nan_count is None:
            nan_col = libcudf.unary.is_nan(self)
            self._nan_count = nan_col.sum()
        return self._nan_count

    def dropna(self, drop_nan: bool = False) -> NumericalColumn:
        col = self.nans_to_nulls() if drop_nan else self
        return drop_nulls([col])[0]

    @property
    def contains_na_entries(self) -> bool:
        return (self.nan_count != 0) or (self.null_count != 0)

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

    def _can_return_nan(self, skipna: bool = None) -> bool:
        return not skipna and self.has_nulls(include_nan=True)

    def _process_for_reduction(
        self, skipna: bool = None, min_count: int = 0
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
                full(len(to_replace_col), replacement[0], self.dtype)
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
        method: str = None,
        dtype: Dtype = None,
        fill_nan: bool = True,
    ) -> NumericalColumn:
        """
        Fill null values with *fill_value*
        """
        col = self.nans_to_nulls() if fill_nan else self

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
                            f"{col.dtype.type.__name__} to "
                            f"{cudf.dtype(dtype).type.__name__}"
                        )
                    fill_value = new_fill_value
            else:
                fill_value = fill_value.astype(col.dtype)

        return super(NumericalColumn, col).fillna(fill_value, method)

    def _find_value(
        self, value: ScalarLike, closest: bool, find: Callable, compare: str
    ) -> int:
        value = to_cudf_compatible_scalar(value)
        if not is_number(value):
            raise ValueError("Expected a numeric value")
        found = 0
        if len(self):
            found = find(
                self.data_array_view,
                value,
                mask=self.mask,
            )
        if found == -1:
            if self.is_monotonic_increasing and closest:
                found = find(
                    self.data_array_view,
                    value,
                    mask=self.mask,
                    compare=compare,
                )
                if found == -1:
                    raise ValueError("value not found")
            else:
                raise ValueError("value not found")
        return found

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        """
        Returns offset of first value that matches. For monotonic
        columns, returns the offset of the first larger value
        if closest=True.
        """
        if self.is_monotonic_increasing and closest:
            if value < self.min():
                return 0
            elif value > self.max():
                return len(self)
        return self._find_value(value, closest, cudautils.find_first, "gt")

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches. For monotonic
        columns, returns the offset of the last smaller value
        if closest=True.
        """
        if self.is_monotonic_increasing and closest:
            if value < self.min():
                return -1
            elif value > self.max():
                return len(self) - 1
        return self._find_value(value, closest, cudautils.find_last, "lt")

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
            iinfo = np.iinfo(to_dtype)
            min_, max_ = iinfo.min, iinfo.max

            # best we can do is hope to catch it here and avoid compare
            if (self.min() >= min_) and (self.max() <= max_):
                filled = self.fillna(0, fill_nan=False)
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
        self, index: pd.Index = None, nullable: bool = False, **kwargs
    ) -> "pd.Series":
        if nullable and self.dtype in np_dtypes_to_pandas_dtypes:
            pandas_nullable_dtype = np_dtypes_to_pandas_dtypes[self.dtype]
            arrow_array = self.to_arrow()
            pandas_array = pandas_nullable_dtype.__from_arrow__(arrow_array)
            pd_series = pd.Series(pandas_array, copy=False)
        elif str(self.dtype) in NUMERIC_TYPES and not self.has_nulls():
            pd_series = pd.Series(cupy.asnumpy(self.values), copy=False)
        else:
            pd_series = self.to_arrow().to_pandas(**kwargs)

        if index is not None:
            pd_series.index = index
        return pd_series

    def _reduction_result_dtype(self, reduction_op: str) -> Dtype:
        col_dtype = self.dtype
        if reduction_op in {"sum", "product"}:
            col_dtype = (
                col_dtype if col_dtype.kind == "f" else np.dtype("int64")
            )
        elif reduction_op == "sum_of_squares":
            col_dtype = np.find_common_type([col_dtype], [np.dtype("uint64")])

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
            else:
                col_to_normalize_casted = input_column_dtype.type(
                    col_to_normalize[0]
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
