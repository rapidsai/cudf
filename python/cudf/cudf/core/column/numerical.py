# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations

from numbers import Number
from typing import Any, Callable, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from nvtx import annotate
from pandas.api.types import is_integer_dtype

import cudf
from cudf import _lib as libcudf
from cudf._lib.quantiles import quantile as cpp_quantile
from cudf._typing import BinaryOperand, ColumnLike, Dtype, DtypeObj, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_column,
    column,
    string,
)
from cudf.utils import cudautils, utils
from cudf.utils.dtypes import (
    min_column_type,
    min_signed_type,
    numeric_normalize_types,
    to_cudf_compatible_scalar,
)


class NumericalColumn(ColumnBase):
    def __init__(
        self,
        data: Buffer,
        dtype: DtypeObj,
        mask: Buffer = None,
        size: int = None,  # TODO: make this non-optional
        offset: int = 0,
        null_count: int = None,
    ):
        """
        Parameters
        ----------
        data : Buffer
        dtype : np.dtype
            The dtype associated with the data Buffer
        mask : Buffer, optional
        """
        dtype = np.dtype(dtype)
        if data.size % dtype.itemsize:
            raise ValueError("Buffer size must be divisible by element size")
        if size is None:
            size = data.size // dtype.itemsize
            size = size - offset

        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
        )

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

    def unary_operator(self, unaryop: str) -> ColumnBase:
        return _numeric_column_unaryop(self, op=unaryop)

    def binary_operator(
        self, binop: str, rhs: BinaryOperand, reflect: bool = False,
    ) -> ColumnBase:
        int_dtypes = [
            np.dtype("int8"),
            np.dtype("int16"),
            np.dtype("int32"),
            np.dtype("int64"),
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("uint32"),
            np.dtype("uint64"),
        ]
        if rhs is None:
            out_dtype = self.dtype
        else:
            if not (
                isinstance(rhs, (NumericalColumn, cudf.Scalar,),)
                or np.isscalar(rhs)
            ):
                msg = "{!r} operator not supported between {} and {}"
                raise TypeError(msg.format(binop, type(self), type(rhs)))
            out_dtype = np.result_type(self.dtype, rhs.dtype)
            if binop in ["mod", "floordiv"]:
                tmp = self if reflect else rhs
                if (tmp.dtype in int_dtypes) and (
                    (np.isscalar(tmp) and (0 == tmp))
                    or ((isinstance(tmp, NumericalColumn)) and (0.0 in tmp))
                ):
                    out_dtype = np.dtype("float64")
        return _numeric_column_binop(
            lhs=self, rhs=rhs, op=binop, out_dtype=out_dtype, reflect=reflect
        )

    def _apply_scan_op(self, op: str) -> ColumnBase:
        return libcudf.reduce.scan(op, self, True)

    def normalize_binop_value(
        self, other: ScalarLike
    ) -> Union[ColumnBase, ScalarLike]:
        if other is None:
            return other
        if isinstance(other, cudf.Scalar):
            if self.dtype == other.dtype:
                return other
            # expensive device-host transfer just to
            # adjust the dtype
            other = other.value
        elif isinstance(other, np.ndarray) and other.ndim == 0:
            other = other.item()
        other_dtype = np.min_scalar_type(other)
        if other_dtype.kind in {"b", "i", "u", "f"}:
            if isinstance(other, cudf.Scalar):
                return other
            other_dtype = np.promote_types(self.dtype, other_dtype)
            if other_dtype == np.dtype("float16"):
                other_dtype = np.dtype("float32")
                other = other_dtype.type(other)
            if self.dtype.kind == "b":
                other_dtype = min_signed_type(other)
            if np.isscalar(other):
                other = np.dtype(other_dtype).type(other)
                return other
            else:
                ary = utils.scalar_broadcast_to(
                    other, size=len(self), dtype=other_dtype
                )
                return column.build_column(
                    data=Buffer(ary), dtype=ary.dtype, mask=self.mask,
                )
        else:
            raise TypeError(f"cannot broadcast {type(other)}")

    def int2ip(self) -> "cudf.core.column.StringColumn":
        if self.dtype != np.dtype("int64"):
            raise TypeError("Only int64 type can be converted to ip")

        return libcudf.string_casting.int2ip(self)

    def as_string_column(
        self, dtype: Dtype, format=None
    ) -> "cudf.core.column.StringColumn":
        if len(self) > 0:
            return string._numeric_to_str_typecast_functions[
                np.dtype(self.dtype)
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
    ) -> "cudf.core.column.DecimalColumn":
        if is_integer_dtype(self.dtype):
            raise NotImplementedError(
                "Casting from integer types to decimal "
                "types not currently supported"
            )
        result = libcudf.unary.cast(self, dtype)
        if isinstance(dtype, cudf.core.dtypes.Decimal64Dtype):
            result.dtype.precision = dtype.precision
        return result

    def as_numerical_column(self, dtype: Dtype) -> NumericalColumn:
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        return libcudf.unary.cast(self, dtype)

    def reduce(self, op: str, skipna: bool = None, **kwargs) -> float:
        min_count = kwargs.pop("min_count", 0)
        preprocessed = self._process_for_reduction(
            skipna=skipna, min_count=min_count
        )
        if isinstance(preprocessed, ColumnBase):
            return libcudf.reduce.reduce(op, preprocessed, **kwargs)
        else:
            return cast(float, preprocessed)

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> float:
        return self.reduce(
            "sum", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def product(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ) -> float:
        return self.reduce(
            "product", skipna=skipna, dtype=dtype, min_count=min_count
        )

    def mean(self, skipna: bool = None, dtype: Dtype = np.float64) -> float:
        return self.reduce("mean", skipna=skipna, dtype=dtype)

    def var(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = np.float64
    ) -> float:
        return self.reduce("var", skipna=skipna, dtype=dtype, ddof=ddof)

    def std(
        self, skipna: bool = None, ddof: int = 1, dtype: Dtype = np.float64
    ) -> float:
        return self.reduce("std", skipna=skipna, dtype=dtype, ddof=ddof)

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

    def sum_of_squares(self, dtype: Dtype = None) -> float:
        return libcudf.reduce.reduce("sum_of_squares", self, dtype=dtype)

    def kurtosis(self, skipna: bool = None) -> float:
        skipna = True if skipna is None else skipna

        if len(self) == 0 or (not skipna and self.has_nulls):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()  # type: ignore

        if len(self) < 4:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m4_numerator = ((self - miu) ** self.normalize_binop_value(4)).sum()
        V = self.var()

        if V == 0:
            return 0

        term_one_section_one = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term_one_section_two = m4_numerator / (V ** 2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        return kurt

    def skew(self, skipna: bool = None) -> ScalarLike:
        skipna = True if skipna is None else skipna

        if len(self) == 0 or (not skipna and self.has_nulls):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()  # type: ignore

        if len(self) < 3:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m3 = (((self - miu) ** self.normalize_binop_value(3)).sum()) / n
        m2 = self.var(ddof=0)

        if m2 == 0:
            return 0

        unbiased_coef = ((n * (n - 1)) ** 0.5) / (n - 2)
        skew = unbiased_coef * m3 / (m2 ** (3 / 2))
        return skew

    def quantile(
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> NumericalColumn:
        if isinstance(q, Number) or cudf.utils.dtypes.is_list_like(q):
            np_array_q = np.asarray(q)
            if np.logical_or(np_array_q < 0, np_array_q > 1).any():
                raise ValueError(
                    "percentiles should all be in the interval [0, 1]"
                )
        # Beyond this point, q either being scalar or list-like
        # will only have values in range [0, 1]
        result = self._numeric_quantile(q, interpolation, exact)
        if isinstance(q, Number):
            return (
                cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
                if result[0] is cudf.NA
                else result[0]
            )
        return result

    def median(self, skipna: bool = None) -> NumericalColumn:
        skipna = True if skipna is None else skipna

        if not skipna and self.has_nulls:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        # enforce linear in case the default ever changes
        return self.quantile(0.5, interpolation="linear", exact=True)

    def _numeric_quantile(
        self, q: Union[float, Sequence[float]], interpolation: str, exact: bool
    ) -> NumericalColumn:
        quant = [float(q)] if not isinstance(q, (Sequence, np.ndarray)) else q
        # get sorted indices and exclude nulls
        sorted_indices = self.as_frame()._get_sorted_inds(True, "first")
        sorted_indices = sorted_indices[self.null_count :]

        return cpp_quantile(self, quant, interpolation, sorted_indices, exact)

    def cov(self, other: ColumnBase) -> float:
        if (
            len(self) == 0
            or len(other) == 0
            or (len(self) == 1 and len(other) == 1)
        ):
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        result = (self - self.mean()) * (other - other.mean())
        cov_sample = result.sum() / (len(self) - 1)
        return cov_sample

    def corr(self, other: ColumnBase) -> float:
        if len(self) == 0 or len(other) == 0:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)

        cov = self.cov(other)
        lhs_std, rhs_std = self.std(), other.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return cudf.utils.dtypes._get_nan_for_dtype(self.dtype)
        return cov / lhs_std / rhs_std

    def round(self, decimals: int = 0) -> NumericalColumn:
        """Round the values in the Column to the given number of decimals.
        """
        return libcudf.round.round(self, decimal_places=decimals)

    def applymap(
        self, udf: Callable[[ScalarLike], ScalarLike], out_dtype: Dtype = None
    ) -> ColumnBase:
        """Apply an element-wise function to transform the values in the Column.

        Parameters
        ----------
        udf : function
            Wrapped by numba jit for call on the GPU as a device function.
        out_dtype  : numpy.dtype; optional
            The dtype for use in the output.
            By default, use the same dtype as *self.dtype*.

        Returns
        -------
        result : Column
            The mask is preserved.
        """
        if out_dtype is None:
            out_dtype = self.dtype
        out = column.column_applymap(udf=udf, column=self, out_dtype=out_dtype)
        return out

    def default_na_value(self) -> ScalarLike:
        """Returns the default NA value for this column
        """
        dkind = self.dtype.kind
        if dkind == "f":
            return self.dtype.type(np.nan)
        elif dkind == "i":
            return np.iinfo(self.dtype).min
        elif dkind == "u":
            return np.iinfo(self.dtype).max
        elif dkind == "b":
            return self.dtype.type(False)
        else:
            raise TypeError(f"numeric column of {self.dtype} has no NaN value")

    def find_and_replace(
        self,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> NumericalColumn:
        """
        Return col with *to_replace* replaced with *value*.
        """
        to_replace_col = as_column(to_replace)
        replacement_col = as_column(replacement)

        if type(to_replace_col) != type(replacement_col):
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
        replaced = self.copy()
        if len(replacement_col) == 1 and len(to_replace_col) > 1:
            replacement_col = column.as_column(
                utils.scalar_broadcast_to(
                    replacement[0], (len(to_replace_col),), self.dtype
                )
            )
        elif len(replacement_col) == 1 and len(to_replace_col) == 0:
            return replaced
        to_replace_col, replacement_col, replaced = numeric_normalize_types(
            to_replace_col, replacement_col, replaced
        )
        return libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
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
        if fill_nan:
            col = self.nans_to_nulls()
        else:
            col = self

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
            # cast safely to the same dtype as self
            if is_integer_dtype(col.dtype):
                fill_value = _safe_cast_to_int(fill_value, col.dtype)
            else:
                fill_value = fill_value.astype(col.dtype)

        return super(NumericalColumn, col).fillna(fill_value, method)

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        """
        Returns offset of first value that matches. For monotonic
        columns, returns the offset of the first larger value
        if closest=True.
        """
        value = to_cudf_compatible_scalar(value)
        if not pd.api.types.is_number(value):
            raise ValueError("Expected a numeric value")
        found = 0
        if len(self):
            found = cudautils.find_first(
                self.data_array_view, value, mask=self.mask
            )
        if found == -1 and self.is_monotonic and closest:
            if value < self.min():
                found = 0
            elif value > self.max():
                found = len(self)
            else:
                found = cudautils.find_first(
                    self.data_array_view, value, mask=self.mask, compare="gt",
                )
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches. For monotonic
        columns, returns the offset of the last smaller value
        if closest=True.
        """
        value = to_cudf_compatible_scalar(value)
        if not pd.api.types.is_number(value):
            raise ValueError("Expected a numeric value")
        found = 0
        if len(self):
            found = cudautils.find_last(
                self.data_array_view, value, mask=self.mask,
            )
        if found == -1 and self.is_monotonic and closest:
            if value < self.min():
                found = -1
            elif value > self.max():
                found = len(self) - 1
            else:
                found = cudautils.find_last(
                    self.data_array_view, value, mask=self.mask, compare="lt",
                )
                if found == -1:
                    raise ValueError("value not found")
        elif found == -1:
            raise ValueError("value not found")
        return found

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
                    info = np.finfo(to_dtype)
                elif "int" in to_dtype.name:
                    info = np.iinfo(to_dtype)
                lower_, upper_ = info.min, info.max

                if self.dtype.kind == "f":
                    # Exclude 'np.inf', '-np.inf'
                    s = cudf.Series(self)
                    # TODO: replace np.inf with cudf scalar when
                    # https://github.com/rapidsai/cudf/pull/6297 merges
                    non_infs = s[
                        ((s == np.inf) | (s == -np.inf)).logical_not()
                    ]
                    col = non_infs._column
                else:
                    col = self

                min_ = col.min()
                # TODO: depending on implementation of cudf scalar and future
                # refactor of min/max, change the test method
                if np.isnan(min_):
                    # Column contains only infs
                    return True

                max_ = col.max()
                if (min_ >= lower_) and (max_ < upper_):
                    return True
                else:
                    return False

        # want to cast int to uint
        elif self.dtype.kind == "i" and to_dtype.kind == "u":
            i_max_ = np.iinfo(self.dtype).max
            u_max_ = np.iinfo(to_dtype).max

            if self.min() >= 0:
                if i_max_ <= u_max_:
                    return True
                if self.max() < u_max_:
                    return True
            return False

        # want to cast uint to int
        elif self.dtype.kind == "u" and to_dtype.kind == "i":
            u_max_ = np.iinfo(self.dtype).max
            i_max_ = np.iinfo(to_dtype).max

            if u_max_ <= i_max_:
                return True
            if self.max() < i_max_:
                return True
            return False

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
                if (
                    cudf.Series(filled).astype(to_dtype).astype(filled.dtype)
                    == cudf.Series(filled)
                ).all():
                    return True
                else:
                    return False

        # want to cast float to int:
        elif self.dtype.kind == "f" and to_dtype.kind in {"i", "u"}:
            info = np.iinfo(to_dtype)
            min_, max_ = info.min, info.max

            # best we can do is hope to catch it here and avoid compare
            if (self.min() >= min_) and (self.max() <= max_):
                filled = self.fillna(0, fill_nan=False)
                if (cudf.Series(filled) % 1 == 0).all():
                    return True
                else:
                    return False
            else:
                return False

        return False


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def _numeric_column_binop(
    lhs: Union[ColumnBase, ScalarLike],
    rhs: Union[ColumnBase, ScalarLike],
    op: str,
    out_dtype: Dtype,
    reflect: bool = False,
) -> ColumnBase:
    if reflect:
        lhs, rhs = rhs, lhs

    is_op_comparison = op in [
        "lt",
        "gt",
        "le",
        "ge",
        "eq",
        "ne",
        "NULL_EQUALS",
    ]

    if is_op_comparison:
        out_dtype = "bool"

    out = libcudf.binaryop.binaryop(lhs, rhs, op, out_dtype)

    return out


def _numeric_column_unaryop(operand: ColumnBase, op: str) -> ColumnBase:
    if callable(op):
        return libcudf.transform.transform(operand, op)

    op = libcudf.unary.UnaryOp[op.upper()]
    return libcudf.unary.unary_operation(operand, op)


def _safe_cast_to_int(col: ColumnBase, dtype: DtypeObj) -> ColumnBase:
    """
    Cast given NumericalColumn to given integer dtype safely.
    """
    assert is_integer_dtype(dtype)

    if col.dtype == dtype:
        return col

    new_col = col.astype(dtype)
    if (new_col == col).all():
        return new_col
    else:
        raise TypeError(
            f"Cannot safely cast non-equivalent "
            f"{col.dtype.type.__name__} to {np.dtype(dtype).type.__name__}"
        )


def _normalize_find_and_replace_input(
    input_column_dtype: DtypeObj, col_to_normalize: Union[ColumnBase, list]
) -> ColumnBase:
    normalized_column = column.as_column(
        col_to_normalize,
        dtype=input_column_dtype if len(col_to_normalize) <= 0 else None,
    )
    col_to_normalize_dtype = normalized_column.dtype
    if isinstance(col_to_normalize, list):
        col_to_normalize_dtype = min_column_type(
            normalized_column, input_column_dtype
        )
        # Scalar case
        if len(col_to_normalize) == 1:
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

    return as_column(
        libcudf.sort.digitize(column.as_frame(), bin_col.as_frame(), right)
    )
