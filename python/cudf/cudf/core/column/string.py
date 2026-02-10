# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import re
import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.dtype.validators import is_dtype_obj_string
from cudf.core.mixins import Scannable
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    get_dtype_of_same_kind,
    is_pandas_nullable_extension_dtype,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import infer_format
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import cupy as cp

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.decimal import DecimalBaseColumn
    from cudf.core.column.lists import ListColumn
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.timedelta import TimeDeltaColumn
    from cudf.core.dtypes import DecimalDtype


# For now all supported re flags have matching names in libcudf. If that ever changes
# this construction will need to be updated with more explicit mapping.
_FLAG_MAP = {
    getattr(re, flag): getattr(plc.strings.regex_flags.RegexFlags, flag)
    for flag in ("MULTILINE", "DOTALL")
}


@lru_cache
def plc_flags_from_re_flags(
    flags: re.RegexFlag,
) -> plc.strings.regex_flags.RegexFlags:
    # Convert Python re flags to pylibcudf RegexFlags
    plc_flags = plc.strings.regex_flags.RegexFlags(0)
    for re_flag, plc_flag in _FLAG_MAP.items():
        if flags & re_flag:
            plc_flags |= plc_flag
            flags &= ~re_flag
    if flags:
        raise ValueError(f"Unsupported re flags: {flags}")
    return plc_flags


class StringColumn(ColumnBase, Scannable):
    """Implements operations for Columns of String type"""

    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        # These operators aren't actually supported, they only exist to allow
        # empty column binops with scalars of arbitrary other dtypes. See
        # the _binaryop method for more information.
        "__sub__",
        "__mul__",
        "__mod__",
        "__pow__",
        "__truediv__",
        "__floordiv__",
    }
    _VALID_PLC_TYPES = {plc.TypeId.STRING}
    _VALID_SCANS = {
        "cummin",
        "cummax",
    }

    @property
    def _PANDAS_NA_VALUE(self) -> ScalarLike:
        """String columns return None as NA value in pandas compatibility mode."""
        if cudf.get_option("mode.pandas_compatible"):
            if is_pandas_nullable_extension_dtype(self.dtype):
                return self.dtype.na_value
            return None
        return pd.NA

    @classmethod
    def _validate_args(
        cls, plc_column: plc.Column, dtype: np.dtype
    ) -> tuple[plc.Column, np.dtype]:
        plc_column, dtype = super()._validate_args(plc_column, dtype)
        if not is_dtype_obj_string(dtype):
            raise ValueError("dtype must be a valid cuDF string dtype")
        return plc_column, dtype

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise NotImplementedError(
            f"dtype {self.dtype} is not yet supported via "
            "`__cuda_array_interface__`"
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if (
            cudf.get_option("mode.pandas_compatible")
            and is_scalar(fill_value)
            and fill_value is np.nan
        ):
            raise MixedTypeError("Cannot fill `np.nan` in string column")
        return super()._validate_fillna_value(fill_value)

    def element_indexing(self, index: int) -> str | None:
        result = super().element_indexing(index)
        if isinstance(result, pa.Scalar):
            return result.as_py()
        return result

    def to_arrow(self) -> pa.Array:
        # All null string columns fail to convert in libcudf, so we must short-circuit
        # the call to super().to_arrow().
        # TODO: Investigate if the above is a bug in libcudf and fix it there.
        if self.plc_column.num_children() == 0 or self.is_all_null:
            return pa.NullArray.from_buffers(
                pa.null(), len(self), [pa.py_buffer(b"")]
            )
        return super().to_arrow()

    def sum(
        self,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        col = self.nans_to_nulls() if skipna else self
        if not skipna and col.has_nulls():
            return pd.NA
        if skipna:
            col = col.dropna()

        if min_count > 0 and col.valid_count < min_count:
            return pd.NA

        return (
            0
            if len(col) == 0
            else col.join_strings("", None).element_indexing(0)
        )

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        """Check if any string value is truthy (non-empty)."""
        if not skipna and self.has_nulls():
            raise TypeError("boolean value of NA is ambiguous")
        elif skipna and self.null_count == self.size:
            return False
        raise NotImplementedError("`any` not implemented for `StringColumn`")

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> ScalarLike:
        """Check if all string values are truthy (non-empty)."""
        if skipna and self.null_count == self.size:
            return True
        elif not skipna and self.has_nulls():
            raise TypeError("boolean value of NA is ambiguous")
        raise NotImplementedError("`all` not implemented for `StringColumn`")

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Validate reduction operations for StringColumn."""
        if op in {"min", "max"}:
            return super()._reduce(op, skipna, min_count, **kwargs)

        # For empty columns with statistical operations, return NaN (pandas behavior)
        if len(self) == 0 and op in {"var", "std", "mean", "sum_of_squares"}:
            return np.nan

        raise TypeError(f"Series.{op} does not support StringColumn")

    def __contains__(self, item: ScalarLike) -> bool:
        other = [item] if is_scalar(item) else item
        return self.contains(as_column(other, dtype=self.dtype)).any()

    def _with_type_metadata(self: Self, dtype: DtypeObj) -> Self:
        """
        Copies type metadata from self onto other, returning a new column.
        """
        # For pandas dtypes, store them directly in the column's dtype property
        if (
            isinstance(dtype, pd.ArrowDtype) and dtype.kind == "U"
        ) or isinstance(dtype, pd.StringDtype):
            self._dtype = dtype
        return self

    def _get_pandas_compatible_dtype(self, target_dtype: np.dtype) -> DtypeObj:
        """
        Get the appropriate dtype for pandas-compatible mode.

        For StringDtype with na_value=np.nan, returns get_dtype_of_same_kind(self.dtype, target_dtype).
        Otherwise, returns get_dtype_of_same_kind(pd.StringDtype() or self.dtype, target_dtype).
        """
        if (
            isinstance(self.dtype, pd.StringDtype)
            and self.dtype.na_value is np.nan
        ):
            return get_dtype_of_same_kind(self.dtype, target_dtype)
        else:
            return get_dtype_of_same_kind(
                pd.StringDtype()
                if isinstance(self.dtype, pd.StringDtype)
                else self.dtype,
                target_dtype,
            )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        if dtype.kind == "b":
            result = self.count_characters() > np.int8(0)
            if not is_pandas_nullable_extension_dtype(dtype):
                result = result.fillna(False)
            return result._with_type_metadata(dtype)

        cast_func: Callable[[plc.Column, plc.DataType], plc.Column]
        if dtype.kind in {"i", "u"}:
            if not self.is_all_integer():
                raise ValueError(
                    "Could not convert strings to integer "
                    "type due to presence of non-integer values."
                )
            cast_func = plc.strings.convert.convert_integers.to_integers
        elif dtype.kind == "f":
            if not self.is_all_float():
                raise ValueError(
                    "Could not convert strings to float "
                    "type due to presence of non-floating values."
                )
            cast_func = plc.strings.convert.convert_floats.to_floats
        else:
            raise ValueError(f"dtype must be a numerical type, not {dtype}")
        plc_dtype = dtype_to_pylibcudf_type(dtype)
        with self.access(mode="read", scope="internal"):
            return cast(
                cudf.core.column.numerical.NumericalColumn,
                ColumnBase.create(
                    cast_func(self.plc_column, plc_dtype), dtype
                ),
            )

    def strptime(
        self, dtype: DtypeObj, format: str
    ) -> DatetimeColumn | TimeDeltaColumn:
        if dtype.kind not in "Mm":
            raise ValueError(
                f"dtype must be datetime or timedelta type, not {dtype}"
            )
        elif self.is_all_null:
            return column_empty(len(self), dtype=dtype)  # type: ignore[return-value]
        elif (self == "None").any():
            raise ValueError(
                "Cannot convert `None` value to datetime or timedelta."
            )

        casting_func: Callable[[plc.Column, plc.DataType, str], plc.Column]
        if dtype.kind == "M":
            if format.endswith("%z"):
                raise NotImplementedError(
                    "cuDF does not yet support timezone-aware datetimes"
                )
            is_nat = self == "NaT"
            without_nat = self.apply_boolean_mask(is_nat.unary_operator("not"))
            char_counts = without_nat.count_characters()  # type: ignore[attr-defined]
            if char_counts.distinct_count(dropna=True) != 1:
                # Unfortunately disables OK cases like:
                # ["2020-01-01", "2020-01-01 00:00:00"]
                # But currently incorrect for cases like (drops 10):
                # ["2020-01-01", "2020-01-01 10:00:00"]
                raise NotImplementedError(
                    "Cannot parse date-like strings with different formats"
                )
            valid_ts = self.is_timestamp(format)
            valid = valid_ts | is_nat
            if not valid.all():
                raise ValueError(f"Column contains invalid data for {format=}")

            casting_func = plc.strings.convert.convert_datetime.to_timestamps
            add_back_nat = is_nat.any()
        elif dtype.kind == "m":
            casting_func = plc.strings.convert.convert_durations.to_durations
            add_back_nat = False

        with self.access(mode="read", scope="internal"):
            plc_dtype = dtype_to_pylibcudf_type(dtype)
            result_col = cast(
                cudf.core.column.datetime.DatetimeColumn
                | cudf.core.column.timedelta.TimeDeltaColumn,
                ColumnBase.create(
                    casting_func(self.plc_column, plc_dtype, format), dtype
                ),
            )

        if add_back_nat:
            result_col[is_nat] = None

        return result_col

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        not_null = self.apply_boolean_mask(self.notnull())
        if len(not_null) == 0:
            # We should hit the self.null_count == len(self) condition
            # so format doesn't matter
            format = ""
        else:
            # infer on host from the first not na element
            format = infer_format(not_null.element_indexing(0))
        return self.strptime(dtype, format)  # type: ignore[return-value]

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return self.strptime(dtype, "%D days %H:%M:%S")  # type: ignore[return-value]

    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalBaseColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = (
                plc.strings.convert.convert_fixed_point.to_fixed_point(
                    self.plc_column,
                    dtype_to_pylibcudf_type(dtype),
                )
            )
            return cast(
                "cudf.core.column.decimal.DecimalBaseColumn",
                ColumnBase.create(plc_column, dtype),
            )

    def as_string_column(self, dtype: DtypeObj) -> Self:
        if dtype != self.dtype:
            return cast(Self, ColumnBase.create(self.plc_column, dtype))
        return self

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the Column.
        """
        # dask checks for a TypeError instead of NotImplementedError
        raise TypeError(f"cupy does not support {self.dtype}")

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(self.dtype, pd.StringDtype)
            and self.dtype.storage in ["pyarrow", "python"]
        ):
            if self.dtype.storage == "pyarrow":
                pandas_array = self.dtype.__from_arrow__(
                    self.to_arrow().cast(pa.large_string())
                )
            elif self.dtype.na_value is np.nan:
                pandas_array = pd.array(
                    self.to_arrow().to_pandas(), dtype=self.dtype
                )
            else:
                return super().to_pandas(
                    nullable=nullable, arrow_type=arrow_type
                )
            return pd.Index(pandas_array, copy=False)
        return super().to_pandas(nullable=nullable, arrow_type=arrow_type)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        if self.dtype == to_dtype:
            return True
        elif to_dtype.kind in {"i", "u"} and self.is_all_integer():
            return True
        elif to_dtype.kind == "f" and self.is_all_float():
            return True
        else:
            return False

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """
        Return col with *to_replace* replaced with *value*
        """

        to_replace_col = as_column(to_replace)
        replacement_col = as_column(replacement)

        if type(to_replace_col) is not type(replacement_col):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )

        if (
            to_replace_col.dtype != self.dtype
            and replacement_col.dtype != self.dtype
        ):
            return self.copy()
        df = cudf.DataFrame._from_data(
            {"old": to_replace_col, "new": replacement_col}
        )
        df = df.drop_duplicates(subset=["old"], keep="last", ignore_index=True)
        if df._data["old"].null_count == 1:
            res = self.fillna(
                df._data["new"]
                .apply_boolean_mask(df._data["old"].isnull())
                .element_indexing(0)
            )
            df = df.dropna(subset=["old"])
        else:
            res = self
        return res.replace(df._data["old"], df._data["new"])

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        # Due to https://github.com/pandas-dev/pandas/issues/46332 we need to
        # support binary operations between empty or all null string columns
        # and columns of other dtypes, even if those operations would otherwise
        # be invalid. For example, you cannot divide strings, but pandas allows
        # division between an empty string column and a (nonempty) integer
        # column. Ideally we would disable these operators entirely, but until
        # the above issue is resolved we cannot avoid this problem.
        if self.is_all_null:
            if op in {
                "__add__",
                "__sub__",
                "__mul__",
                "__mod__",
                "__pow__",
                "__truediv__",
                "__floordiv__",
            }:
                return self
            elif op in {"__eq__", "__lt__", "__le__", "__gt__", "__ge__"}:
                return self.notnull()
            elif op == "__ne__":
                return self.isnull()

        if is_scalar(other):
            if is_na_like(other):
                other = pa.scalar(None, type=cudf_dtype_to_pa_type(self.dtype))
            else:
                other = pa.scalar(other)  # type: ignore[arg-type]
        elif not isinstance(other, type(self)):
            return NotImplemented

        if isinstance(other, (StringColumn, pa.Scalar)):
            if isinstance(other, pa.Scalar) and not pa.types.is_string(
                other.type
            ):
                if op in {"__eq__", "__ne__"}:
                    return as_column(
                        op == "__ne__",
                        length=len(self),
                        dtype=get_dtype_of_same_kind(
                            self.dtype, np.dtype(np.bool_)
                        ),
                    ).set_mask(self.mask, self.null_count)
                else:
                    return NotImplemented

            if op == "__add__":
                if isinstance(other, pa.Scalar):
                    other = cast(
                        cudf.core.column.string.StringColumn,
                        as_column(other, length=len(self)),
                    )
                lhs, rhs = (other, self) if reflect else (self, other)
                return lhs.concatenate([rhs], "", None)._with_type_metadata(
                    self.dtype
                )
            elif op in {
                "__eq__",
                "__ne__",
                "__gt__",
                "__lt__",
                "__ge__",
                "__le__",
                "NULL_EQUALS",
                "NULL_NOT_EQUALS",
            }:
                if isinstance(other, pa.Scalar):
                    other = pa_scalar_to_plc_scalar(other)
                lhs_op, rhs_op = (other, self) if reflect else (self, other)
                return binaryop.binaryop(
                    lhs=lhs_op,
                    rhs=rhs_op,
                    op=op,
                    dtype=get_dtype_of_same_kind(
                        self.dtype, np.dtype(np.bool_)
                    ),
                )
        return NotImplemented

    def minhash(
        self,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            result = plc.nvtext.minhash.minhash(
                self.plc_column,
                seed,
                a.plc_column,
                b.plc_column,
                width,
            )
            return cast(
                cudf.core.column.lists.ListColumn,
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def minhash64(
        self,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint64 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint64).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                    )
                seed = np.uint64(seed)
            result = plc.nvtext.minhash.minhash64(
                self.plc_column,
                seed,
                a.plc_column,
                b.plc_column,
                width,
            )
            return cast(
                cudf.core.column.lists.ListColumn,
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint64))
                    ),
                ),
            )

    def jaccard_index(self, other: Self, width: int) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.jaccard.jaccard_index(
                self.plc_column,
                other.plc_column,
                width,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.float32)),
                ),
            )

    def generate_ngrams(self, ngrams: int, separator: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.generate_ngrams.generate_ngrams(
                self.plc_column,
                ngrams,
                separator,
            )
            return cast(
                Self,
                ColumnBase.create(result, self.dtype),
            )

    def generate_character_ngrams(self, ngrams: int) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.generate_ngrams.generate_character_ngrams(
                self.plc_column, ngrams
            )
            return cast(
                "cudf.core.column.lists.ListColumn",
                ColumnBase.create(result, cudf.ListDtype(self.dtype)),
            )

    def hash_character_ngrams(
        self, ngrams: int, seed: int | np.uint32
    ) -> ListColumn:
        with self.access(mode="read", scope="internal"):
            # Convert int to np.uint32 with validation
            if isinstance(seed, int):
                if seed < 0 or seed > np.iinfo(np.uint32).max:
                    raise ValueError(
                        f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                    )
                seed = np.uint32(seed)
            result = plc.nvtext.generate_ngrams.hash_character_ngrams(
                self.plc_column, ngrams, seed
            )
            return cast(
                cudf.core.column.lists.ListColumn,
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32))
                    ),
                ),
            )

    def build_suffix_array(self, min_width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.build_suffix_array(
                self.plc_column, min_width
            )
            return cast(
                Self,
                ColumnBase.create(result, np.dtype(np.int32)),
            )

    def resolve_duplicates(self, sa: Self, min_width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.resolve_duplicates(
                self.plc_column,
                sa.plc_column,
                min_width,
            )
            return cast(
                Self,
                ColumnBase.create(result, self.dtype),
            )

    def resolve_duplicates_pair(
        self, sa1: Self, input2: Self, sa2: Self, min_width: int
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.deduplicate.resolve_duplicates_pair(
                self.plc_column,
                sa1.plc_column,
                input2.plc_column,
                sa2.plc_column,
                min_width,
            )
            return cast(
                Self,
                ColumnBase.create(result, self.dtype),
            )

    def edit_distance(self, targets: Self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.edit_distance.edit_distance(
                self.plc_column, targets.plc_column
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def edit_distance_matrix(self) -> ListColumn:
        warnings.warn(
            "edit_distance_matrix is deprecated. Use edit_distance instead.",
            FutureWarning,
        )
        with self.access(mode="read", scope="internal"):
            result = plc.nvtext.edit_distance.edit_distance_matrix(
                self.plc_column
            )
            return cast(
                cudf.core.column.lists.ListColumn,
                ColumnBase.create(
                    result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def byte_pair_encoding(
        self,
        merge_pairs: plc.nvtext.byte_pair_encode.BPEMergePairs,
        separator: str,
    ) -> Self:
        warnings.warn(
            "byte_pair_encoding is deprecated and will be removed in a future version.",
            FutureWarning,
            stacklevel=2,
        )
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.byte_pair_encode.byte_pair_encoding(
                self.plc_column,
                merge_pairs,
                pa_scalar_to_plc_scalar(pa.scalar(separator)),
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def ngrams_tokenize(
        self,
        ngrams: int,
        delimiter: plc.Scalar,
        separator: plc.Scalar,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.ngrams_tokenize.ngrams_tokenize(
                self.plc_column,
                ngrams,
                delimiter,
                separator,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def normalize_spaces(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.normalize.normalize_spaces(self.plc_column)
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def normalize_characters(
        self, normalizer: plc.nvtext.normalize.CharacterNormalizer
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.normalize.normalize_characters(
                self.plc_column,
                normalizer,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_tokens(
        self, targets: Self, replacements: Self, delimiter: plc.Scalar
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.replace.replace_tokens(
                self.plc_column,
                targets.plc_column,
                replacements.plc_column,
                delimiter,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def filter_tokens(
        self,
        min_token_length: int,
        replacement: plc.Scalar,
        delimiter: plc.Scalar,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.replace.filter_tokens(
                self.plc_column,
                min_token_length,
                replacement,
                delimiter,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def porter_stemmer_measure(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.stemmer.porter_stemmer_measure(
                self.plc_column
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def is_letter(
        self, is_vowel: bool, index: int | NumericalColumn
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.stemmer.is_letter(
                self.plc_column,
                is_vowel,
                index if isinstance(index, int) else index.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def tokenize_scalar(self, delimiter: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.tokenize_scalar(
                self.plc_column, delimiter
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def tokenize_column(self, delimiters: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.tokenize_column(
                self.plc_column,
                delimiters.plc_column,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def count_tokens_scalar(self, delimiter: plc.Scalar) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.count_tokens_scalar(
                self.plc_column, delimiter
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def count_tokens_column(self, delimiters: Self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.count_tokens_column(
                self.plc_column,
                delimiters.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def character_tokenize(self) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.nvtext.tokenize.character_tokenize(self.plc_column),
                    cudf.ListDtype(self.dtype),
                ),
            )

    def tokenize_with_vocabulary(
        self,
        vocabulary: plc.nvtext.tokenize.TokenizeVocabulary,
        delimiter: str,
        default_id: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.nvtext.tokenize.tokenize_with_vocabulary(
                        self.plc_column,
                        vocabulary,
                        pa_scalar_to_plc_scalar(pa.scalar(delimiter)),
                        default_id,
                    ),
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def wordpiece_tokenize(
        self,
        vocabulary: plc.nvtext.wordpiece_tokenize.WordPieceVocabulary,
        max_words_per_row: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.nvtext.wordpiece_tokenize.wordpiece_tokenize(
                        self.plc_column,
                        vocabulary,
                        max_words_per_row,
                    ),
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def detokenize(self, indices: ColumnBase, separator: plc.Scalar) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.nvtext.tokenize.detokenize(
                self.plc_column,
                indices.plc_column,
                separator,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def _modify_characters(
        self, method: Callable[[plc.Column], plc.Column]
    ) -> Self:
        """Helper function for methods that modify characters e.g. to_lower"""
        with self.access(mode="read", scope="internal"):
            plc_column = method(self.plc_column)
            return cast(Self, ColumnBase.create(plc_column, self.dtype))

    def to_lower(self) -> Self:
        with self.access(mode="read", scope="internal"):
            result = cast(
                Self,
                ColumnBase.create(
                    plc.strings.case.to_lower(self.plc_column), self.dtype
                ),
            )

        # Handle Greek final sigma (ς) special case in pandas compatibility mode
        # Greek capital sigma (Σ) lowercases to regular sigma (σ) at libcudf level,  # noqa: RUF003
        # but should become final sigma (ς) when at the end of a word.
        # Replace σ with ς when followed by end-of-string or non-letter character.  # noqa: RUF003
        if cudf.get_option("mode.pandas_compatible"):
            has_sigma = result.str_contains("σ")
            if has_sigma.any():
                result = result.replace_with_backrefs(
                    r"σ($|[^a-zA-Zα-ωΑ-Ωά-ώΆ-Ώ])", r"ς\1"
                )

        return result

    def to_upper(self) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.strings.case.to_upper(self.plc_column), self.dtype
                ),
            )

    def capitalize(self) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.strings.capitalize.capitalize(self.plc_column),
                    self.dtype,
                ),
            )

    def swapcase(self) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.strings.case.swapcase(self.plc_column), self.dtype
                ),
            )

    def title(self) -> Self:
        with self.access(mode="read", scope="internal"):
            return cast(
                Self,
                ColumnBase.create(
                    plc.strings.capitalize.title(self.plc_column), self.dtype
                ),
            )

    def is_title(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.capitalize.is_title(self.plc_column)
            return cast(
                Self,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def replace_multiple(self, pattern: Self, replacements: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace_multiple(
                self.plc_column,
                pattern.plc_column,
                replacements.plc_column,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def is_hex(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_integers.is_hex(
                self.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def hex_to_integers(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_integers.hex_to_integers(
                self.plc_column, plc.DataType(plc.TypeId.INT64)
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int64)),
                ),
            )

    def is_ipv4(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_ipv4.is_ipv4(
                self.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def ipv4_to_integers(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_ipv4.ipv4_to_integers(
                self.plc_column,
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.uint32)),
                ),
            )

    def is_timestamp(self, format: str) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_datetime.is_timestamp(
                self.plc_column, format
            )
            return cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype("bool")),
                ),
            )

    def _split_record_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Column,
        ],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                maxsplit,
            )
            result_dtype = get_dtype_of_same_kind(
                self.dtype, cudf.ListDtype(self.dtype)
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, result_dtype),
            )

    def split_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.split_record_re
        )

    def rsplit_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.rsplit_record_re
        )

    def _split_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Table,
        ],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                maxsplit,
            )
            return dict(
                enumerate(
                    cast(
                        Self,
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def split_re(self, pattern: str, maxsplit: int) -> dict[int, Self]:
        return self._split_re(
            pattern, maxsplit, plc.strings.split.split.split_re
        )

    def rsplit_re(self, pattern: str, maxsplit: int) -> dict[int, Self]:
        return self._split_re(
            pattern, maxsplit, plc.strings.split.split.rsplit_re
        )

    def _split_record(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Column],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = method(
                self.plc_column,
                delimiter,
                maxsplit,
            )
            result_dtype = get_dtype_of_same_kind(
                self.dtype, cudf.ListDtype(self.dtype)
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, result_dtype),
            )

    def split_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.split_record
        )

    def rsplit_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.rsplit_record
        )

    def _split(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Table],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                delimiter,
                maxsplit,
            )
            return dict(
                enumerate(
                    cast(
                        Self,
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def split(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.split)

    def rsplit(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.rsplit)

    def split_part(self, delimiter: plc.Scalar, index: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.split.split.split_part(
                self.plc_column,
                delimiter,
                index,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def _partition(
        self,
        delimiter: plc.Scalar,
        method: Callable[[plc.Column, plc.Scalar], plc.Table],
    ) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = method(
                self.plc_column,
                delimiter,
            )
            return dict(
                enumerate(
                    cast(
                        Self,
                        ColumnBase.create(col, self.dtype),
                    )
                    for col in plc_table.columns()
                )
            )

    def partition(self, delimiter: plc.Scalar) -> dict[int, Self]:
        return self._partition(
            delimiter, plc.strings.split.partition.partition
        )

    def rpartition(self, delimiter: plc.Scalar) -> dict[int, Self]:
        return self._partition(
            delimiter, plc.strings.split.partition.rpartition
        )

    def url_decode(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_urls.url_decode(
                self.plc_column
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def url_encode(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_urls.url_encode(
                self.plc_column
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def is_integer(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_integers.is_integer(
                self.plc_column
            )
            return cast(
                cudf.core.column.numerical.NumericalColumn,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def is_float(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.convert.convert_floats.is_float(
                self.plc_column
            )
            return cast(
                cudf.core.column.numerical.NumericalColumn,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def is_all_integer(self) -> bool:
        """Check if all non-null strings in the column are integers.

        This is an optimized version of `is_integer().all()` that avoids
        creating an intermediate boolean column.

        Returns
        -------
        bool
            True if all non-null strings are valid integers, False otherwise.
        """
        with self.access(mode="read", scope="internal"):
            bool_plc = plc.strings.convert.convert_integers.is_integer(
                self.plc_column
            )
            return self._reduce_bool_column(bool_plc)

    def is_all_float(self) -> bool:
        """Check if all non-null strings in the column are floats.

        This is an optimized version of `is_float().all()` that avoids
        creating an intermediate boolean column.

        Returns
        -------
        bool
            True if all non-null strings are valid floats, False otherwise.
        """
        with self.access(mode="read", scope="internal"):
            bool_plc = plc.strings.convert.convert_floats.is_float(
                self.plc_column
            )
            return self._reduce_bool_column(bool_plc)

    @staticmethod
    def _reduce_bool_column(bool_plc: plc.Column) -> bool:
        """Reduce a boolean column to a single bool using all()."""
        result_scalar = plc.reduce.reduce(
            bool_plc,
            plc.aggregation.all(),
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )
        result = result_scalar.to_py()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        return result

    def count_characters(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.attributes.count_characters(
                self.plc_column
            )
            dtype = self._get_pandas_compatible_dtype(np.dtype(np.int32))
            res = ColumnBase.create(
                plc_column,
                dtype,
            )
            return cast(cudf.core.column.numerical.NumericalColumn, res)

    def count_bytes(self) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.attributes.count_bytes(self.plc_column)
            return cast(
                cudf.core.column.numerical.NumericalColumn,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def join_strings(self, separator: str, na_rep: str | None) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.combine.join_strings(
                self.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar(separator)),
                pa_scalar_to_plc_scalar(pa.scalar(na_rep, type=pa.string())),
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def concatenate(
        self, others: Iterable[Self], sep: str, na_rep: str | None
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.combine.concatenate(
                plc.Table(
                    [col.plc_column for col in itertools.chain([self], others)]
                ),
                pa_scalar_to_plc_scalar(pa.scalar(sep)),
                pa_scalar_to_plc_scalar(pa.scalar(na_rep, type=pa.string())),
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def extract(self, pattern: str, flags: int) -> dict[int, Self]:
        with self.access(mode="read", scope="internal"):
            plc_table = plc.strings.extract.extract(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc_flags_from_re_flags(flags),
                ),
            )
            return dict(
                enumerate(
                    cast(Self, ColumnBase.create(col, self.dtype))
                    for col in plc_table.columns()
                )
            )

    def contains_re(self, pattern: str, flags: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.contains.contains_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc_flags_from_re_flags(flags),
                ),
            )
            dtype = self._get_pandas_compatible_dtype(np.dtype(np.bool_))
            return cast(Self, ColumnBase.create(plc_column, dtype))

    def str_contains(self, pattern: str | Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_pattern = (
                pa_scalar_to_plc_scalar(pa.scalar(pattern))
                if isinstance(pattern, str)
                else pattern.plc_column
            )
            plc_column = plc.strings.find.contains(
                self.plc_column,
                plc_pattern,
            )
            return cast(
                Self,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def like(self, pattern: str, escape: str) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.contains.like(
                self.plc_column,
                pattern,
                escape,
            )
            return cast(
                Self,
                ColumnBase.create(
                    plc_column,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_)),
                ),
            )

    def repeat_strings(self, repeats: int | ColumnBase) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_repeats = (
                repeats.plc_column
                if isinstance(repeats, ColumnBase)
                else repeats
            )
            plc_column = plc.strings.repeat.repeat_strings(
                self.plc_column,
                plc_repeats,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_re(
        self,
        pattern: list[str] | str,
        replacement: Self | pa.Scalar,
        max_replace_count: int = -1,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            if isinstance(pattern, list) and isinstance(
                replacement, type(self)
            ):
                plc_column = plc.strings.replace_re.replace_re(
                    self.plc_column,
                    pattern,
                    replacement.plc_column,
                    max_replace_count,
                )
            elif isinstance(pattern, str) and isinstance(
                replacement, pa.Scalar
            ):
                plc_column = plc.strings.replace_re.replace_re(
                    self.plc_column,
                    plc.strings.regex_program.RegexProgram.create(
                        pattern,
                        plc.strings.regex_flags.RegexFlags.DEFAULT,
                    ),
                    pa_scalar_to_plc_scalar(replacement),
                    max_replace_count,
                )
            else:
                raise ValueError("Invalid pattern and replacement types")
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_str(
        self, pattern: str, replacement: pa.Scalar, max_replace_count: int = -1
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace(
                self.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar(pattern)),
                pa_scalar_to_plc_scalar(replacement),
                max_replace_count,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def replace_with_backrefs(self, pattern: str, replacement: str) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace_re.replace_with_backrefs(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
                ),
                replacement,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def slice_strings(
        self,
        start: int | None | NumericalColumn,
        stop: int | None | NumericalColumn,
        step: int | None = None,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            if isinstance(start, ColumnBase) and isinstance(stop, ColumnBase):
                plc_start: plc.Column | plc.Scalar = start.plc_column
                plc_stop: plc.Column | plc.Scalar = stop.plc_column
                plc_step: plc.Scalar | None = None
            elif all(isinstance(x, int) or x is None for x in (start, stop)):
                param_dtype = pa.int32()
                plc_start = pa_scalar_to_plc_scalar(
                    pa.scalar(start, type=param_dtype)
                )
                plc_stop = pa_scalar_to_plc_scalar(
                    pa.scalar(stop, type=param_dtype)
                )
                plc_step = pa_scalar_to_plc_scalar(
                    pa.scalar(step, type=param_dtype)
                )
            else:
                raise ValueError("Invalid start and stop types")
            plc_result = plc.strings.slice.slice_strings(
                self.plc_column, plc_start, plc_stop, plc_step
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def all_characters_of_type(
        self,
        char_type: plc.strings.char_types.StringCharacterTypes,
        case_type: plc.strings.char_types.StringCharacterTypes = plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    ) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.char_types.all_characters_of_type(
                self.plc_column, char_type, case_type
            )
            dtype = self._get_pandas_compatible_dtype(np.dtype(np.bool_))
            result = cast(
                "cudf.core.column.numerical.NumericalColumn",
                ColumnBase.create(plc_result, dtype),
            )
            if (
                isinstance(self.dtype, pd.StringDtype)
                and self.dtype.na_value is np.nan
            ):
                result = result.fillna(False)
            return result

    def filter_characters_of_type(
        self,
        types_to_remove: plc.strings.char_types.StringCharacterTypes,
        replacement: str,
        types_to_keep: plc.strings.char_types.StringCharacterTypes,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_column = plc.strings.char_types.filter_characters_of_type(
                self.plc_column,
                types_to_remove,
                pa_scalar_to_plc_scalar(
                    pa.scalar(replacement, type=pa.string())
                ),
                types_to_keep,
            )
            return cast(
                Self,
                ColumnBase.create(plc_column, self.dtype),
            )

    def replace_slice(self, start: int, stop: int, repl: str) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.replace.replace_slice(
                self.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar(repl, type=pa.string())),
                start,
                stop,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def get_json_object(
        self,
        json_path: str,
        allow_single_quotes: bool,
        strip_quotes_from_single_strings: bool,
        missing_fields_as_nulls: bool,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            options = plc.json.GetJsonObjectOptions(
                allow_single_quotes=allow_single_quotes,
                strip_quotes_from_single_strings=(
                    strip_quotes_from_single_strings
                ),
                missing_fields_as_nulls=missing_fields_as_nulls,
            )
            plc_result = plc.json.get_json_object(
                self.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar(json_path)),
                options,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def pad(
        self, width: int, side: plc.strings.side_type.SideType, fillchar: str
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.padding.pad(
                self.plc_column,
                width,
                side,
                fillchar,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def zfill(self, width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.padding.zfill(
                self.plc_column,
                width,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def strip(
        self, side: plc.strings.side_type.SideType, to_strip: str | None = None
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.strip.strip(
                self.plc_column,
                side,
                pa_scalar_to_plc_scalar(
                    pa.scalar(to_strip or "", type=pa.string())
                ),
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def wrap(self, width: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.wrap.wrap(
                self.plc_column,
                width,
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def count_re(self, pattern: str, flags: int) -> NumericalColumn:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.contains.count_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc_flags_from_re_flags(flags)
                ),
            )
            res = ColumnBase.create(
                plc_result,
                self._get_pandas_compatible_dtype(np.dtype(np.int32)),
            )
            return cast(cudf.core.column.numerical.NumericalColumn, res)

    def findall(
        self,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram], plc.Column
        ],
        pat: str,
        flags: int = 0,
    ) -> Self:
        # Return type depends on method parameter at runtime:
        # - plc.strings.findall.findall -> LIST<STRING>
        # - plc.strings.findall.find_re -> INT32
        with self.access(mode="read", scope="internal"):
            if len(self) == 0:
                return cast(
                    Self,
                    as_column([], dtype=np.dtype("object")),
                )
            plc_result = method(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pat, plc_flags_from_re_flags(flags)
                ),
            )
            res = type(self).from_pylibcudf(plc_result)
            res = res._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, res.dtype)
            )
            return cast(Self, res)

    def find_multiple(self, patterns: Self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.find_multiple.find_multiple(
                self.plc_column,
                patterns.plc_column,
            )
            return cast(
                Self,
                ColumnBase.create(
                    plc_result,
                    cudf.ListDtype(
                        get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
                    ),
                ),
            )

    def starts_ends_with(
        self,
        method: Callable[[plc.Column, plc.Column | plc.Scalar], plc.Column],
        pat: str | tuple[str, ...],
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            if isinstance(pat, str):
                plc_pat = pa_scalar_to_plc_scalar(
                    pa.scalar(pat, type=pa.string())
                )
                plc_result = method(self.plc_column, plc_pat)
            elif isinstance(pat, tuple) and all(
                isinstance(p, str) for p in pat
            ):
                plc_self = self.plc_column
                plc_pat = pa_scalar_to_plc_scalar(
                    pa.scalar(pat[0], type=pa.string())
                )
                plc_result = method(plc_self, plc_pat)
                for next_pat in pat[1:]:
                    plc_pat = pa_scalar_to_plc_scalar(
                        pa.scalar(next_pat, type=pa.string())
                    )
                    plc_next_result = method(plc_self, plc_pat)
                    plc_result = plc.binaryop.binary_operation(
                        plc_result,
                        plc_next_result,
                        plc.binaryop.BinaryOperator.BITWISE_OR,
                        plc.DataType(plc.TypeId.BOOL8),
                    )
            else:
                raise TypeError(
                    f"expected a str or tuple[str, ...], not {type(pat).__name__}"
                )
            dtype = self._get_pandas_compatible_dtype(np.dtype(np.bool_))
            return cast(Self, ColumnBase.create(plc_result, dtype))

    def find(
        self,
        method: Callable[[plc.Column, plc.Scalar, int, int], plc.Column],
        sub: str,
        start: int,
        end: int,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = method(
                self.plc_column,
                pa_scalar_to_plc_scalar(pa.scalar(sub, type=pa.string())),
                start,
                end,
            )
            base_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.int32))
            target_dtype = base_dtype
            if cudf.get_option("mode.pandas_compatible"):
                target_dtype = self._get_pandas_compatible_dtype(
                    np.dtype("int64")
                )
            if target_dtype != base_dtype:
                plc_result = plc.unary.cast(
                    plc_result, dtype_to_pylibcudf_type(np.dtype("int64"))
                )
            return cast(Self, ColumnBase.create(plc_result, target_dtype))

    def matches_re(self, pattern: str, flags: int) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.contains.matches_re(
                self.plc_column,
                plc.strings.regex_program.RegexProgram.create(
                    pattern, plc_flags_from_re_flags(flags)
                ),
            )
            dtype = self._get_pandas_compatible_dtype(np.dtype(np.bool_))
            return cast(Self, ColumnBase.create(plc_result, dtype))

    def code_points(self) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.attributes.code_points(
                self.plc_column,
            )
            return cast(
                Self,
                ColumnBase.create(
                    plc_result,
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int32)),
                ),
            )

    def translate(self, table: dict) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.translate.translate(
                self.plc_column,
                str.maketrans(table),  # type: ignore[arg-type]
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )

    def filter_characters(
        self,
        table: dict,
        keep: bool = True,
        repl: str | None = None,
    ) -> Self:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.strings.translate.filter_characters(
                self.plc_column,
                str.maketrans(table),  # type: ignore[arg-type]
                plc.strings.translate.FilterType.KEEP
                if keep
                else plc.strings.translate.FilterType.REMOVE,
                pa_scalar_to_plc_scalar(pa.scalar(repl, type=pa.string())),
            )
            return cast(
                Self,
                ColumnBase.create(plc_result, self.dtype),
            )
