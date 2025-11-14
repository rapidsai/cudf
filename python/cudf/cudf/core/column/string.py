# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import re
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Self

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core._internals import binaryop
from cudf.core.buffer import Buffer, acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.mixins import Scannable
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    CUDF_STRING_DTYPE,
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    get_dtype_of_same_kind,
    is_dtype_obj_string,
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
        Dtype,
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
    """
    Implements operations for Columns of String type

    Parameters
    ----------
    data : Buffer
        Buffer of the string data
    mask : Buffer
        The validity mask
    offset : int
        Data offset
    children : Tuple[Column]
        Columns containing the offsets
    """

    _start_offset: int | None
    _end_offset: int | None

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

    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: np.dtype,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        if (
            not cudf.get_option("mode.pandas_compatible")
            and dtype != CUDF_STRING_DTYPE
            and dtype.kind != "U"
        ) or (
            cudf.get_option("mode.pandas_compatible")
            and not is_dtype_obj_string(dtype)
        ):
            raise ValueError(f"dtype must be {CUDF_STRING_DTYPE}")
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(dtype, np.dtype)
            and dtype.kind == "U"
        ):
            dtype = CUDF_STRING_DTYPE

        super().__init__(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )

        self._start_offset = None
        self._end_offset = None

    @property
    def start_offset(self) -> int:
        if self._start_offset is None:
            if (
                len(self.base_children) == 1
                and self.offset < self.base_children[0].size
            ):
                self._start_offset = int(
                    self.base_children[0].element_indexing(self.offset)
                )
            else:
                self._start_offset = 0

        return self._start_offset

    @property
    def end_offset(self) -> int:
        if self._end_offset is None:
            if (
                len(self.base_children) == 1
                and (self.offset + self.size) < self.base_children[0].size
            ):
                self._end_offset = int(
                    self.base_children[0].element_indexing(
                        self.offset + self.size
                    )
                )
            else:
                self._end_offset = 0

        return self._end_offset

    @cached_property
    def memory_usage(self) -> int:
        n = super().memory_usage
        if len(self.base_children) == 1:
            child0_size = (self.size + 1) * self.base_children[
                0
            ].dtype.itemsize

            n += child0_size
        return n

    @property
    def base_size(self) -> int:
        if len(self.base_children) == 0:
            return 0
        else:
            return self.base_children[0].size - 1

    @property
    def data(self) -> None | Buffer:
        if self._data is None:
            assert self.base_data is not None
            if (
                self.offset == 0
                and len(self.base_children) > 0
                and self.size == self.base_children[0].size - 1
            ):
                self._data = self.base_data  # type: ignore[assignment]
            else:
                self._data = self.base_data[  # type: ignore[assignment]
                    self.start_offset : self.end_offset
                ]
        return self._data

    def all(self, skipna: bool = True) -> bool:
        if skipna and self.null_count == self.size:
            return True
        elif not skipna and self.has_nulls():
            raise TypeError("boolean value of NA is ambiguous")
        raise NotImplementedError("`all` not implemented for `StringColumn`")

    def any(self, skipna: bool = True) -> bool:
        if not skipna and self.has_nulls():
            raise TypeError("boolean value of NA is ambiguous")
        elif skipna and self.null_count == self.size:
            return False

        raise NotImplementedError("`any` not implemented for `StringColumn`")

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
        """Convert to PyArrow Array

        Examples
        --------
        >>> import cudf
        >>> col = cudf.core.as_column([1, 2, 3, 4])
        >>> col.to_arrow()
        <pyarrow.lib.Int64Array object at 0x7f886547f830>
        [
          1,
          2,
          3,
          4
        ]
        """
        if self.null_count == len(self):
            return pa.NullArray.from_buffers(
                pa.null(), len(self), [pa.py_buffer(b"")]
            )
        else:
            return super().to_arrow()

    def sum(
        self,
        skipna: bool = True,
        min_count: int = 0,
    ) -> ScalarLike:
        result_col = self._process_for_reduction(
            skipna=skipna, min_count=min_count
        )
        if isinstance(result_col, type(self)):
            return result_col.join_strings("", None).element_indexing(0)
        else:
            return result_col

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

    def _scan(self, op: str):
        return self.scan(op.replace("cum", ""), True)._with_type_metadata(
            self.dtype
        )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        if dtype.kind == "b":
            result = self.count_characters() > np.int8(0)
            if not is_pandas_nullable_extension_dtype(dtype):
                result = result.fillna(False)
            return result._with_type_metadata(dtype)

        cast_func: Callable[[plc.Column, plc.DataType], plc.Column]
        if dtype.kind in {"i", "u"}:
            if not self.is_integer().all():
                raise ValueError(
                    "Could not convert strings to integer "
                    "type due to presence of non-integer values."
                )
            cast_func = plc.strings.convert.convert_integers.to_integers
        elif dtype.kind == "f":
            if not self.is_float().all():
                raise ValueError(
                    "Could not convert strings to float "
                    "type due to presence of non-floating values."
                )
            cast_func = plc.strings.convert.convert_floats.to_floats
        else:
            raise ValueError(f"dtype must be a numerical type, not {dtype}")
        plc_dtype = dtype_to_pylibcudf_type(dtype)
        with acquire_spill_lock():
            return (
                type(self)
                .from_pylibcudf(  # type: ignore[return-value]
                    cast_func(self.to_pylibcudf(mode="read"), plc_dtype)
                )
                ._with_type_metadata(dtype=dtype)
            )

    def strptime(
        self, dtype: Dtype, format: str
    ) -> DatetimeColumn | TimeDeltaColumn:
        if dtype.kind not in "Mm":  # type: ignore[union-attr]
            raise ValueError(
                f"dtype must be datetime or timedelta type, not {dtype}"
            )
        elif self.null_count == len(self):
            return column_empty(len(self), dtype=dtype)  # type: ignore[return-value]
        elif (self == "None").any():
            raise ValueError(
                "Cannot convert `None` value to datetime or timedelta."
            )

        casting_func: Callable[[plc.Column, plc.DataType, str], plc.Column]
        if dtype.kind == "M":  # type: ignore[union-attr]
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
        elif dtype.kind == "m":  # type: ignore[union-attr]
            casting_func = plc.strings.convert.convert_durations.to_durations
            add_back_nat = False

        with acquire_spill_lock():
            plc_dtype = dtype_to_pylibcudf_type(dtype)
            result_col = type(self).from_pylibcudf(
                casting_func(self.to_pylibcudf(mode="read"), plc_dtype, format)
            )

        if add_back_nat:
            result_col[is_nat] = None

        return result_col  # type: ignore[return-value]

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

    @acquire_spill_lock()
    def as_decimal_column(self, dtype: DecimalDtype) -> DecimalBaseColumn:
        plc_column = plc.strings.convert.convert_fixed_point.to_fixed_point(
            self.to_pylibcudf(mode="read"),
            dtype_to_pylibcudf_type(dtype),
        )
        result = ColumnBase.from_pylibcudf(plc_column)
        result.dtype.precision = dtype.precision  # type: ignore[union-attr]
        return result  # type: ignore[return-value]

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        col = self
        if dtype != self.dtype:
            if isinstance(dtype, pd.StringDtype) or (
                isinstance(dtype, pd.ArrowDtype)
                and pa.string() == dtype.pyarrow_dtype
            ):
                # TODO: Drop the deep copies on astype's copy keyword
                # default value is fixed in `25.10`
                col = self.copy(deep=True)
                col._dtype = dtype
            elif isinstance(dtype, np.dtype) and dtype.kind in {"U", "O"}:
                # TODO: Drop the deep copies on astype's copy keyword
                # default value is fixed in `25.10`
                col = self.copy(deep=True)
                col._dtype = CUDF_STRING_DTYPE
        return col

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
        elif to_dtype.kind in {"i", "u"} and self.is_integer().all():
            return True
        elif to_dtype.kind == "f" and self.is_float().all():
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
        if self.null_count == len(self):
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
                    ).set_mask(self.mask)
                else:
                    return NotImplemented

            if op == "__add__":
                if isinstance(other, pa.Scalar):
                    other = cast(
                        StringColumn,
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

    @acquire_spill_lock()
    def minhash(
        self,
        seed: int | np.uint32,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        # Convert int to np.uint32 with validation
        if isinstance(seed, int):
            if seed < 0 or seed > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                )
            seed = np.uint32(seed)
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.minhash.minhash(
                self.to_pylibcudf(mode="read"),
                seed,
                a.to_pylibcudf(mode="read"),
                b.to_pylibcudf(mode="read"),
                width,
            )
        )

    @acquire_spill_lock()
    def minhash64(
        self,
        seed: int | np.uint64,
        a: NumericalColumn,
        b: NumericalColumn,
        width: int,
    ) -> ListColumn:
        # Convert int to np.uint64 with validation
        if isinstance(seed, int):
            if seed < 0 or seed > np.iinfo(np.uint64).max:
                raise ValueError(
                    f"seed must be in range [0, {np.iinfo(np.uint64).max}]"
                )
            seed = np.uint64(seed)
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.minhash.minhash64(
                self.to_pylibcudf(mode="read"),
                seed,
                a.to_pylibcudf(mode="read"),
                b.to_pylibcudf(mode="read"),
                width,
            )
        )

    @acquire_spill_lock()
    def jaccard_index(self, other: Self, width: int) -> NumericalColumn:
        result = plc.nvtext.jaccard.jaccard_index(
            self.to_pylibcudf(mode="read"),
            other.to_pylibcudf(mode="read"),
            width,
        )
        return type(self).from_pylibcudf(result)  # type: ignore[return-value]

    @acquire_spill_lock()
    def generate_ngrams(self, ngrams: int, separator: plc.Scalar) -> Self:
        result = plc.nvtext.generate_ngrams.generate_ngrams(
            self.to_pylibcudf(mode="read"),
            ngrams,
            separator,
        )
        return type(self).from_pylibcudf(result)

    @acquire_spill_lock()
    def generate_character_ngrams(self, ngrams: int) -> ListColumn:
        result = plc.nvtext.generate_ngrams.generate_character_ngrams(
            self.to_pylibcudf(mode="read"), ngrams
        )
        return type(self).from_pylibcudf(result)  # type: ignore[return-value]

    @acquire_spill_lock()
    def hash_character_ngrams(
        self, ngrams: int, seed: int | np.uint32
    ) -> ListColumn:
        # Convert int to np.uint32 with validation
        if isinstance(seed, int):
            if seed < 0 or seed > np.iinfo(np.uint32).max:
                raise ValueError(
                    f"seed must be in range [0, {np.iinfo(np.uint32).max}]"
                )
            seed = np.uint32(seed)
        result = plc.nvtext.generate_ngrams.hash_character_ngrams(
            self.to_pylibcudf(mode="read"), ngrams, seed
        )
        return type(self).from_pylibcudf(result)  # type: ignore[return-value]

    @acquire_spill_lock()
    def build_suffix_array(self, min_width: int) -> Self:
        result = plc.nvtext.deduplicate.build_suffix_array(
            self.to_pylibcudf(mode="read"), min_width
        )
        return type(self).from_pylibcudf(result)

    @acquire_spill_lock()
    def resolve_duplicates(self, sa: Self, min_width: int) -> Self:
        result = plc.nvtext.deduplicate.resolve_duplicates(
            self.to_pylibcudf(mode="read"),
            sa.to_pylibcudf(mode="read"),
            min_width,
        )
        return type(self).from_pylibcudf(result)

    @acquire_spill_lock()
    def resolve_duplicates_pair(
        self, sa1: Self, input2: Self, sa2: Self, min_width: int
    ) -> Self:
        result = plc.nvtext.deduplicate.resolve_duplicates_pair(
            self.to_pylibcudf(mode="read"),
            sa1.to_pylibcudf(mode="read"),
            input2.to_pylibcudf(mode="read"),
            sa2.to_pylibcudf(mode="read"),
            min_width,
        )
        return type(self).from_pylibcudf(result)

    @acquire_spill_lock()
    def edit_distance(self, targets: Self) -> NumericalColumn:
        result = plc.nvtext.edit_distance.edit_distance(
            self.to_pylibcudf(mode="read"), targets.to_pylibcudf(mode="read")
        )
        return type(self).from_pylibcudf(result)  # type: ignore[return-value]

    @acquire_spill_lock()
    def edit_distance_matrix(self) -> ListColumn:
        result = plc.nvtext.edit_distance.edit_distance_matrix(
            self.to_pylibcudf(mode="read")
        )
        return type(self).from_pylibcudf(result)  # type: ignore[return-value]

    @acquire_spill_lock()
    def byte_pair_encoding(
        self,
        merge_pairs: plc.nvtext.byte_pair_encode.BPEMergePairs,
        separator: str,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.byte_pair_encode.byte_pair_encoding(
                self.to_pylibcudf(mode="read"),
                merge_pairs,
                pa_scalar_to_plc_scalar(pa.scalar(separator)),
            )
        )

    @acquire_spill_lock()
    def ngrams_tokenize(
        self,
        ngrams: int,
        delimiter: plc.Scalar,
        separator: plc.Scalar,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.ngrams_tokenize.ngrams_tokenize(
                self.to_pylibcudf(mode="read"),
                ngrams,
                delimiter,
                separator,
            )
        )

    @acquire_spill_lock()
    def normalize_spaces(self) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.normalize.normalize_spaces(
                self.to_pylibcudf(mode="read")
            )
        )

    @acquire_spill_lock()
    def normalize_characters(
        self, normalizer: plc.nvtext.normalize.CharacterNormalizer
    ) -> Self:
        return ColumnBase.from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.normalize.normalize_characters(
                self.to_pylibcudf(mode="read"),
                normalizer,
            )
        )

    @acquire_spill_lock()
    def replace_tokens(
        self, targets: Self, replacements: Self, delimiter: plc.Scalar
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.replace.replace_tokens(
                self.to_pylibcudf(mode="read"),
                targets.to_pylibcudf(mode="read"),
                replacements.to_pylibcudf(mode="read"),
                delimiter,
            )
        )

    @acquire_spill_lock()
    def filter_tokens(
        self,
        min_token_length: int,
        replacement: plc.Scalar,
        delimiter: plc.Scalar,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.replace.filter_tokens(
                self.to_pylibcudf(mode="read"),
                min_token_length,
                replacement,
                delimiter,
            )
        )

    @acquire_spill_lock()
    def porter_stemmer_measure(self) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.stemmer.porter_stemmer_measure(
                self.to_pylibcudf(mode="read")
            )
        )

    @acquire_spill_lock()
    def is_letter(self, is_vowel: bool, index: int | NumericalColumn) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.stemmer.is_letter(
                self.to_pylibcudf(mode="read"),
                is_vowel,
                index
                if isinstance(index, int)
                else index.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def tokenize_scalar(self, delimiter: plc.Scalar) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.tokenize.tokenize_scalar(
                self.to_pylibcudf(mode="read"), delimiter
            )
        )

    @acquire_spill_lock()
    def tokenize_column(self, delimiters: Self) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.tokenize.tokenize_column(
                self.to_pylibcudf(mode="read"),
                delimiters.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def count_tokens_scalar(self, delimiter: plc.Scalar) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.tokenize.count_tokens_scalar(
                self.to_pylibcudf(mode="read"), delimiter
            )
        )

    @acquire_spill_lock()
    def count_tokens_column(self, delimiters: Self) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.nvtext.tokenize.count_tokens_column(
                self.to_pylibcudf(mode="read"),
                delimiters.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def character_tokenize(self) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.tokenize.character_tokenize(
                self.to_pylibcudf(mode="read")
            )
        )

    @acquire_spill_lock()
    def tokenize_with_vocabulary(
        self,
        vocabulary: plc.nvtext.tokenize.TokenizeVocabulary,
        delimiter: str,
        default_id: int,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.tokenize.tokenize_with_vocabulary(
                self.to_pylibcudf(mode="read"),
                vocabulary,
                pa_scalar_to_plc_scalar(pa.scalar(delimiter)),
                default_id,
            )
        )

    @acquire_spill_lock()
    def wordpiece_tokenize(
        self,
        vocabulary: plc.nvtext.wordpiece_tokenize.WordPieceVocabulary,
        max_words_per_row: int,
    ) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.wordpiece_tokenize.wordpiece_tokenize(
                self.to_pylibcudf(mode="read"),
                vocabulary,
                max_words_per_row,
            )
        )

    @acquire_spill_lock()
    def detokenize(self, indices: ColumnBase, separator: plc.Scalar) -> Self:
        return type(self).from_pylibcudf(
            plc.nvtext.tokenize.detokenize(
                self.to_pylibcudf(mode="read"),
                indices.to_pylibcudf(mode="read"),
                separator,
            )
        )

    @acquire_spill_lock()
    def _modify_characters(
        self, method: Callable[[plc.Column], plc.Column]
    ) -> Self:
        """
        Helper function for methods that modify characters e.g. to_lower
        """
        plc_column = method(self.to_pylibcudf(mode="read"))
        return cast(Self, ColumnBase.from_pylibcudf(plc_column))

    def to_lower(self) -> Self:
        return self._modify_characters(
            plc.strings.case.to_lower
        )._with_type_metadata(self.dtype)

    def to_upper(self) -> Self:
        return self._modify_characters(
            plc.strings.case.to_upper
        )._with_type_metadata(self.dtype)

    def capitalize(self) -> Self:
        return self._modify_characters(
            plc.strings.capitalize.capitalize
        )._with_type_metadata(self.dtype)

    def swapcase(self) -> Self:
        return self._modify_characters(
            plc.strings.case.swapcase
        )._with_type_metadata(self.dtype)

    def title(self) -> Self:
        return self._modify_characters(
            plc.strings.capitalize.title
        )._with_type_metadata(self.dtype)

    def is_title(self) -> Self:
        return self._modify_characters(
            plc.strings.capitalize.is_title
        )._with_type_metadata(
            get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
        )

    @acquire_spill_lock()
    def replace_multiple(self, pattern: Self, replacements: Self) -> Self:
        plc_result = plc.strings.replace.replace_multiple(
            self.to_pylibcudf(mode="read"),
            pattern.to_pylibcudf(mode="read"),
            replacements.to_pylibcudf(mode="read"),
        )
        return cast(
            Self,
            ColumnBase.from_pylibcudf(plc_result)._with_type_metadata(
                self.dtype
            ),
        )

    @acquire_spill_lock()
    def is_hex(self) -> NumericalColumn:
        return (
            type(self)
            .from_pylibcudf(  # type: ignore[return-value]
                plc.strings.convert.convert_integers.is_hex(
                    self.to_pylibcudf(mode="read"),
                )
            )
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def hex_to_integers(self) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.strings.convert.convert_integers.hex_to_integers(
                self.to_pylibcudf(mode="read"), plc.DataType(plc.TypeId.INT64)
            )
        )

    @acquire_spill_lock()
    def is_ipv4(self) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.strings.convert.convert_ipv4.is_ipv4(
                self.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def ipv4_to_integers(self) -> NumericalColumn:
        return type(self).from_pylibcudf(  # type: ignore[return-value]
            plc.strings.convert.convert_ipv4.ipv4_to_integers(
                self.to_pylibcudf(mode="read"),
            )
        )

    @acquire_spill_lock()
    def is_timestamp(self, format: str) -> NumericalColumn:
        return (
            type(self)
            .from_pylibcudf(  # type: ignore[return-value]
                plc.strings.convert.convert_datetime.is_timestamp(
                    self.to_pylibcudf(mode="read"), format
                )
            )
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def _split_record_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Column,
        ],
    ) -> Self:
        plc_column = method(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern,
                plc.strings.regex_flags.RegexFlags.DEFAULT,
            ),
            maxsplit,
        )
        res_col = ColumnBase.from_pylibcudf(plc_column)
        return res_col._with_type_metadata(  # type: ignore[return-value]
            get_dtype_of_same_kind(self.dtype, res_col.dtype)
        )

    def split_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.split_record_re
        )

    def rsplit_record_re(self, pattern: str, maxsplit: int) -> Self:
        return self._split_record_re(
            pattern, maxsplit, plc.strings.split.split.rsplit_record_re
        )

    @acquire_spill_lock()
    def _split_re(
        self,
        pattern: str,
        maxsplit: int,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram, int],
            plc.Table,
        ],
    ) -> dict[int, Self]:
        plc_table = method(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern,
                plc.strings.regex_flags.RegexFlags.DEFAULT,
            ),
            maxsplit,
        )
        return dict(
            enumerate(
                ColumnBase.from_pylibcudf(col)._with_type_metadata(self.dtype)  # type: ignore[misc]
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

    @acquire_spill_lock()
    def _split_record(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Column],
    ) -> Self:
        plc_column = method(
            self.to_pylibcudf(mode="read"),
            delimiter,
            maxsplit,
        )
        res_col = type(self).from_pylibcudf(plc_column)
        return res_col._with_type_metadata(
            get_dtype_of_same_kind(self.dtype, res_col.dtype)
        )

    def split_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.split_record
        )

    def rsplit_record(self, delimiter: plc.Scalar, maxsplit: int) -> Self:
        return self._split_record(
            delimiter, maxsplit, plc.strings.split.split.rsplit_record
        )

    @acquire_spill_lock()
    def _split(
        self,
        delimiter: plc.Scalar,
        maxsplit: int,
        method: Callable[[plc.Column, plc.Scalar, int], plc.Table],
    ) -> dict[int, Self]:
        plc_table = method(
            self.to_pylibcudf(mode="read"),
            delimiter,
            maxsplit,
        )
        return dict(
            enumerate(
                ColumnBase.from_pylibcudf(col)._with_type_metadata(self.dtype)  # type: ignore[misc]
                for col in plc_table.columns()
            )
        )

    def split(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.split)

    def rsplit(self, delimiter: plc.Scalar, maxsplit: int) -> dict[int, Self]:
        return self._split(delimiter, maxsplit, plc.strings.split.split.rsplit)

    @acquire_spill_lock()
    def _partition(
        self,
        delimiter: plc.Scalar,
        method: Callable[[plc.Column, plc.Scalar], plc.Table],
    ) -> dict[int, Self]:
        plc_table = method(
            self.to_pylibcudf(mode="read"),
            delimiter,
        )
        return dict(
            enumerate(
                ColumnBase.from_pylibcudf(col)._with_type_metadata(self.dtype)  # type: ignore[misc]
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

    @acquire_spill_lock()
    def url_decode(self) -> Self:
        plc_column = plc.strings.convert.convert_urls.url_decode(
            self.to_pylibcudf(mode="read")
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def url_encode(self) -> Self:
        plc_column = plc.strings.convert.convert_urls.url_encode(
            self.to_pylibcudf(mode="read")
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def is_integer(self) -> NumericalColumn:
        plc_column = plc.strings.convert.convert_integers.is_integer(
            self.to_pylibcudf(mode="read")
        )
        return (
            type(self)  # type: ignore[return-value]
            .from_pylibcudf(plc_column)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def is_float(self) -> NumericalColumn:
        plc_column = plc.strings.convert.convert_floats.is_float(
            self.to_pylibcudf(mode="read")
        )
        return (
            type(self)  # type: ignore[return-value]
            .from_pylibcudf(plc_column)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def count_characters(self) -> NumericalColumn:
        plc_column = plc.strings.attributes.count_characters(
            self.to_pylibcudf(mode="read")
        )
        res = type(self).from_pylibcudf(plc_column)
        if cudf.get_option("mode.pandas_compatible"):
            res = res.astype(  # type: ignore[assignment]
                get_dtype_of_same_kind(self.dtype, np.dtype("int64"))
            )
        else:
            res = res._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, res.dtype)
            )
        return res  # type: ignore[return-value]

    @acquire_spill_lock()
    def count_bytes(self) -> NumericalColumn:
        plc_column = plc.strings.attributes.count_bytes(
            self.to_pylibcudf(mode="read")
        )
        res = type(self).from_pylibcudf(plc_column)
        res = res._with_type_metadata(
            get_dtype_of_same_kind(self.dtype, res.dtype)
        )
        return res  # type: ignore[return-value]

    @acquire_spill_lock()
    def join_strings(self, separator: str, na_rep: str | None) -> Self:
        plc_column = plc.strings.combine.join_strings(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(pa.scalar(separator)),
            pa_scalar_to_plc_scalar(pa.scalar(na_rep, type=pa.string())),
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def concatenate(
        self, others: Iterable[Self], sep: str, na_rep: str | None
    ) -> Self:
        plc_column = plc.strings.combine.concatenate(
            plc.Table(
                [
                    col.to_pylibcudf(mode="read")
                    for col in itertools.chain([self], others)
                ]
            ),
            pa_scalar_to_plc_scalar(pa.scalar(sep)),
            pa_scalar_to_plc_scalar(pa.scalar(na_rep, type=pa.string())),
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def extract(self, pattern: str, flags: int) -> dict[int, Self]:
        plc_table = plc.strings.extract.extract(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern,
                plc_flags_from_re_flags(flags),
            ),
        )
        return dict(
            enumerate(
                type(self).from_pylibcudf(col)._with_type_metadata(self.dtype)
                for col in plc_table.columns()
            )
        )

    @acquire_spill_lock()
    def contains_re(self, pattern: str, flags: int) -> Self:
        plc_column = plc.strings.contains.contains_re(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern,
                plc_flags_from_re_flags(flags),
            ),
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def str_contains(self, pattern: str | Self) -> Self:
        plc_pattern = (
            pa_scalar_to_plc_scalar(pa.scalar(pattern))
            if isinstance(pattern, str)
            else pattern.to_pylibcudf(mode="read")
        )
        plc_column = plc.strings.find.contains(
            self.to_pylibcudf(mode="read"),
            plc_pattern,
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def like(self, pattern: str, escape: str) -> Self:
        plc_column = plc.strings.contains.like(
            self.to_pylibcudf(mode="read"),
            pattern,
            escape,
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def repeat_strings(self, repeats: int | ColumnBase) -> Self:
        plc_repeats = (
            repeats.to_pylibcudf(mode="read")
            if isinstance(repeats, ColumnBase)
            else repeats
        )
        plc_column = plc.strings.repeat.repeat_strings(
            self.to_pylibcudf(mode="read"),
            plc_repeats,
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def replace_re(
        self,
        pattern: list[str] | str,
        replacement: Self | pa.Scalar,
        max_replace_count: int = -1,
    ) -> Self:
        if isinstance(pattern, list) and isinstance(replacement, type(self)):
            plc_column = plc.strings.replace_re.replace_re(
                self.to_pylibcudf(mode="read"),
                pattern,
                replacement.to_pylibcudf(mode="read"),
                max_replace_count,
            )
        elif isinstance(pattern, str) and isinstance(replacement, pa.Scalar):
            plc_column = plc.strings.replace_re.replace_re(
                self.to_pylibcudf(mode="read"),
                plc.strings.regex_program.RegexProgram.create(
                    pattern,
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                pa_scalar_to_plc_scalar(replacement),
                max_replace_count,
            )
        else:
            raise ValueError("Invalid pattern and replacement types")
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def replace_str(
        self, pattern: str, replacement: pa.Scalar, max_replace_count: int = -1
    ) -> Self:
        plc_result = plc.strings.replace.replace(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(pa.scalar(pattern)),
            pa_scalar_to_plc_scalar(replacement),
            max_replace_count,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def replace_with_backrefs(self, pattern: str, replacement: str) -> Self:
        plc_result = plc.strings.replace_re.replace_with_backrefs(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern, plc.strings.regex_flags.RegexFlags.DEFAULT
            ),
            replacement,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def slice_strings(
        self,
        start: int | None | NumericalColumn,
        stop: int | None | NumericalColumn,
        step: int | None = None,
    ) -> Self:
        if isinstance(start, ColumnBase) and isinstance(stop, ColumnBase):
            plc_start: plc.Column | plc.Scalar = start.to_pylibcudf(
                mode="read"
            )
            plc_stop: plc.Column | plc.Scalar = stop.to_pylibcudf(mode="read")
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
            self.to_pylibcudf(mode="read"), plc_start, plc_stop, plc_step
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def all_characters_of_type(
        self,
        char_type: plc.strings.char_types.StringCharacterTypes,
        case_type: plc.strings.char_types.StringCharacterTypes = plc.strings.char_types.StringCharacterTypes.ALL_TYPES,
    ) -> NumericalColumn:
        plc_result = plc.strings.char_types.all_characters_of_type(
            self.to_pylibcudf(mode="read"), char_type, case_type
        )
        return (
            type(self)  # type: ignore[return-value]
            .from_pylibcudf(plc_result)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def filter_characters_of_type(
        self,
        types_to_remove: plc.strings.char_types.StringCharacterTypes,
        replacement: str,
        types_to_keep: plc.strings.char_types.StringCharacterTypes,
    ) -> Self:
        plc_column = plc.strings.char_types.filter_characters_of_type(
            self.to_pylibcudf(mode="read"),
            types_to_remove,
            pa_scalar_to_plc_scalar(pa.scalar(replacement, type=pa.string())),
            types_to_keep,
        )
        return (
            type(self)
            .from_pylibcudf(plc_column)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def replace_slice(self, start: int, stop: int, repl: str) -> Self:
        plc_result = plc.strings.replace.replace_slice(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(pa.scalar(repl, type=pa.string())),
            start,
            stop,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def get_json_object(
        self,
        json_path: str,
        allow_single_quotes: bool,
        strip_quotes_from_single_strings: bool,
        missing_fields_as_nulls: bool,
    ) -> Self:
        options = plc.json.GetJsonObjectOptions(
            allow_single_quotes=allow_single_quotes,
            strip_quotes_from_single_strings=(
                strip_quotes_from_single_strings
            ),
            missing_fields_as_nulls=missing_fields_as_nulls,
        )
        plc_result = plc.json.get_json_object(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(pa.scalar(json_path)),
            options,
        )
        return type(self).from_pylibcudf(plc_result)

    @acquire_spill_lock()
    def pad(
        self, width: int, side: plc.strings.side_type.SideType, fillchar: str
    ) -> Self:
        plc_result = plc.strings.padding.pad(
            self.to_pylibcudf(mode="read"),
            width,
            side,
            fillchar,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def zfill(self, width: int) -> Self:
        plc_result = plc.strings.padding.zfill(
            self.to_pylibcudf(mode="read"),
            width,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def strip(
        self, side: plc.strings.side_type.SideType, to_strip: str | None = None
    ) -> Self:
        plc_result = plc.strings.strip.strip(
            self.to_pylibcudf(mode="read"),
            side,
            pa_scalar_to_plc_scalar(
                pa.scalar(to_strip or "", type=pa.string())
            ),
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def wrap(self, width: int) -> Self:
        plc_result = plc.strings.wrap.wrap(
            self.to_pylibcudf(mode="read"),
            width,
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def count_re(self, pattern: str, flags: int) -> NumericalColumn:
        plc_result = plc.strings.contains.count_re(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern, plc_flags_from_re_flags(flags)
            ),
        )
        res = type(self).from_pylibcudf(plc_result)
        if is_pandas_nullable_extension_dtype(self.dtype) and not isinstance(
            self.dtype, pd.ArrowDtype
        ):
            res = res.astype(
                get_dtype_of_same_kind(self.dtype, np.dtype("int64"))
            )  # type: ignore[assignment]
        else:
            res = res._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, res.dtype)
            )
        return res  # type: ignore[return-value]

    @acquire_spill_lock()
    def findall(
        self,
        method: Callable[
            [plc.Column, plc.strings.regex_program.RegexProgram], plc.Column
        ],
        pat: str,
        flags: int = 0,
    ) -> Self:
        plc_result = method(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pat, plc_flags_from_re_flags(flags)
            ),
        )
        res = type(self).from_pylibcudf(plc_result)
        res = res._with_type_metadata(
            get_dtype_of_same_kind(self.dtype, res.dtype)
        )
        return res

    @acquire_spill_lock()
    def find_multiple(self, patterns: Self) -> Self:
        plc_result = plc.strings.find_multiple.find_multiple(
            self.to_pylibcudf(mode="read"),
            patterns.to_pylibcudf(mode="read"),
        )
        return type(self).from_pylibcudf(plc_result)

    @acquire_spill_lock()
    def starts_ends_with(
        self,
        method: Callable[[plc.Column, plc.Column | plc.Scalar], plc.Column],
        pat: str | tuple[str, ...],
    ) -> Self:
        if isinstance(pat, str):
            plc_pat = pa_scalar_to_plc_scalar(pa.scalar(pat, type=pa.string()))
            plc_result = method(self.to_pylibcudf(mode="read"), plc_pat)
        elif isinstance(pat, tuple) and all(isinstance(p, str) for p in pat):
            plc_self = self.to_pylibcudf(mode="read")
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
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            )
        )

    @acquire_spill_lock()
    def find(
        self,
        method: Callable[[plc.Column, plc.Scalar, int, int], plc.Column],
        sub: str,
        start: int,
        end: int,
    ) -> Self:
        plc_result = method(
            self.to_pylibcudf(mode="read"),
            pa_scalar_to_plc_scalar(pa.scalar(sub, type=pa.string())),
            start,
            end,
        )
        res = type(self).from_pylibcudf(plc_result)
        if not is_pandas_nullable_extension_dtype(self.dtype) and (
            end is None or end == -1
        ):
            res = res._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, res.dtype)
            )
        else:
            res = res.astype(np.dtype("int64"))  # type: ignore[assignment]
            res = res._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("int64"))
            )
        return res

    @acquire_spill_lock()
    def matches_re(self, pattern: str, flags: int) -> Self:
        plc_result = plc.strings.contains.matches_re(
            self.to_pylibcudf(mode="read"),
            plc.strings.regex_program.RegexProgram.create(
                pattern, plc_flags_from_re_flags(flags)
            ),
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(
                get_dtype_of_same_kind(self.dtype, np.dtype("bool"))
            )
        )

    @acquire_spill_lock()
    def code_points(self) -> Self:
        plc_result = plc.strings.attributes.code_points(
            self.to_pylibcudf(mode="read"),
        )
        res = type(self).from_pylibcudf(plc_result)
        res = res._with_type_metadata(
            get_dtype_of_same_kind(self.dtype, res.dtype)
        )
        return res

    @acquire_spill_lock()
    def translate(self, table: dict) -> Self:
        plc_result = plc.strings.translate.translate(
            self.to_pylibcudf(mode="read"),
            str.maketrans(table),  # type: ignore[arg-type]
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )

    @acquire_spill_lock()
    def filter_characters(
        self,
        table: dict,
        keep: bool = True,
        repl: str | None = None,
    ) -> Self:
        plc_result = plc.strings.translate.filter_characters(
            self.to_pylibcudf(mode="read"),
            str.maketrans(table),  # type: ignore[arg-type]
            plc.strings.translate.FilterType.KEEP
            if keep
            else plc.strings.translate.FilterType.REMOVE,
            pa_scalar_to_plc_scalar(pa.scalar(repl, type=pa.string())),
        )
        return (
            type(self)
            .from_pylibcudf(plc_result)
            ._with_type_metadata(self.dtype)
        )
