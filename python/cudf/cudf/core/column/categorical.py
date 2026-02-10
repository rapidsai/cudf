# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core.column.column import (
    ColumnBase,
    as_column,
    column_empty,
    concat_columns,
)
from cudf.core.column.utils import access_columns
from cudf.core.dtypes import CategoricalDtype, IntervalDtype
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    find_common_type,
    is_mixed_with_object_dtype,
    min_signed_type,
    min_unsigned_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import is_na_like

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableSequence, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column import (
        DatetimeColumn,
        NumericalColumn,
        StringColumn,
        TimeDeltaColumn,
    )


# Using np.int8(-1) to allow silent wrap-around when casting to uint
# it may make sense to make this dtype specific or a function.
_DEFAULT_CATEGORICAL_VALUE = np.int8(-1)


def _sort_column(col: ColumnBase) -> ColumnBase:
    """Sort a column in ascending order with nulls after."""
    with col.access(mode="read", scope="internal"):
        table = plc.Table([col.plc_column])
        sorted_table = plc.sorting.sort(
            table,
            column_order=[plc.types.Order.ASCENDING],
            null_precedence=[plc.types.NullOrder.AFTER],
        )
    return ColumnBase.create(sorted_table.columns()[0], col.dtype)


class CategoricalColumn(ColumnBase):
    """Implements operations for Columns of Categorical type"""

    dtype: CategoricalDtype
    _VALID_REDUCTIONS = {
        "max",
        "min",
    }
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
    }
    # TODO: See if we can narrow these integer types
    _VALID_PLC_TYPES = {
        plc.TypeId.INT8,
        plc.TypeId.INT16,
        plc.TypeId.INT32,
        plc.TypeId.INT64,
        plc.TypeId.UINT8,
        plc.TypeId.UINT16,
        plc.TypeId.UINT32,
        plc.TypeId.UINT64,
    }

    @staticmethod
    def _validate_dtype_to_plc_column(
        plc_column: plc.Column, dtype: DtypeObj
    ) -> None:
        """Validate that the dtype matches the equivalent type of the plc_column"""
        return None

    @classmethod
    def _validate_args(  # type: ignore[override]
        cls, plc_column: plc.Column, dtype: CategoricalDtype
    ) -> tuple[plc.Column, CategoricalDtype]:
        plc_column, dtype = super()._validate_args(plc_column, dtype)  # type: ignore[assignment]
        if not isinstance(dtype, CategoricalDtype):
            raise ValueError(f"{dtype=} must be a CategoricalDtype instance")
        return plc_column, dtype

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            encoded = self._encode(item)
        except ValueError:
            return False
        return encoded in self.codes

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
        # Convert values to categorical dtype like self
        return self, as_column(values, dtype=self.dtype)

    @property
    def categories(self) -> ColumnBase:
        return self.dtype.categories._column

    @cached_property
    def codes(self) -> NumericalColumn:
        """The integer codes representing each category.

        This is a NumericalColumn wrapping self.plc_column, which is necessary because
        many operations on categoricals need to delegate to the codes column.
        """
        return cudf.core.column.NumericalColumn._from_preprocessed(
            self.plc_column, self.dtype._codes_dtype
        )

    @property
    def ordered(self) -> bool | None:
        return self.dtype.ordered

    def __setitem__(self, key: Any, value: Any) -> None:
        if is_scalar(value) and is_na_like(value):
            to_add_categories = 0
        else:
            length = 1 if is_scalar(value) else None
            arr = as_column(value, length=length, nan_as_null=False)
            to_add_categories = len(
                cudf.Index._from_column(arr).difference(
                    cudf.Index._from_column(self.categories)
                )
            )

        if to_add_categories > 0:
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        if is_scalar(value):
            value = self._encode(value) if value is not None else value
        else:
            value = as_column(value).astype(self.dtype)
            value = value.codes
        codes = self.codes
        codes[key] = value
        self.plc_column = codes.plc_column
        self._clear_cache()

    def _fill(
        self,
        fill_value: plc.Scalar,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self | None:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        fill_code = self._encode(fill_value.to_arrow())
        fill_code_scalar = pa_scalar_to_plc_scalar(pa.scalar(fill_code))

        return super()._fill(fill_code_scalar, begin, end, inplace)

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Custom reduction for categorical columns - delegates to codes."""
        # Only valid reductions are min and max
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op}. "
                "You can use .as_ordered() to change the Categorical "
                "to an ordered one."
            )

        # Delegate to underlying codes column via public method
        result = getattr(self.codes, op)(
            skipna=skipna, min_count=min_count, **kwargs
        )

        # Decode the result back to a category
        return self._decode(result)

    def all(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> bool:
        """Categorical columns don't support all() reduction."""
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction 'all'"
        )

    def any(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> bool:
        """Categorical columns don't support any() reduction."""
        raise TypeError(
            f"'{type(self).__name__}' with dtype {self.dtype} "
            f"does not support reduction 'any'"
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        if isinstance(other, ColumnBase):
            if isinstance(other, CategoricalColumn):
                if self.dtype == other.dtype:
                    # Dtypes are the same, but ordering may differ
                    # in which case we need to align them for their codes
                    # to match.
                    if not self.dtype._internal_eq(other.dtype):
                        other = other.as_categorical_column(self.dtype)
                else:
                    raise TypeError(
                        "Categoricals can only compare with the same type"
                    )
            # We'll compare self's decategorized values later for non-CategoricalColumn
        else:
            encoded = self._encode(other)
            if isinstance(encoded, np.generic):
                encoded = encoded.item()
            plc_scalar = plc.Scalar.from_py(
                encoded,
                dtype_to_pylibcudf_type(self.dtype._codes_dtype),
            )
            plc_col = plc.Column.from_scalar(plc_scalar, len(self))
            other = cast(
                CategoricalColumn,
                ColumnBase.create(plc_col, self.dtype),
            )
        equality_ops = {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}
        if not self.ordered and op not in equality_ops:
            raise TypeError(
                "The only binary operations supported by unordered "
                "categorical columns are equality and inequality."
            )
        if not isinstance(other, CategoricalColumn):
            if op not in equality_ops:
                raise TypeError(
                    f"Cannot compare a Categorical for op {op} with a "
                    "non-categorical type. If you want to compare values, "
                    "decategorize the Categorical first."
                )
            elif op not in equality_ops.union(
                {"__gt__", "__lt__", "__ge__", "__le__"}
            ):
                # TODO: Other non-comparison ops may raise or be supported
                return NotImplemented
            return self._get_decategorized_column()._binaryop(other, op)
        return self.codes._binaryop(other.codes, op)

    def element_indexing(self, index: int) -> ScalarLike:
        val = super().element_indexing(index)
        if val is self._PANDAS_NA_VALUE:
            return val
        return self._decode(val.as_py())

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise TypeError(
            "Categorical does not support `__cuda_array_interface__`."
            " Please consider using `.codes` or `.categories`"
            " if you need this functionality."
        )

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if nullable:
            raise NotImplementedError(f"{nullable=} is not supported.")
        if arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not supported.")

        if self.categories.dtype.kind == "f":
            new_mask, null_count = self.notnull().fillna(False).as_mask()
            col = self.set_mask(new_mask, null_count)
        else:
            col = self

        signed_dtype = min_signed_type(len(col.categories))
        codes = (
            col.codes.astype(signed_dtype)
            .fillna(_DEFAULT_CATEGORICAL_VALUE)
            .to_numpy()
        )

        cats = col.categories.nans_to_nulls()
        if not isinstance(cats.dtype, IntervalDtype):
            # leaving out dropna because it temporarily changes an interval
            # index into a struct and throws off results.
            # TODO: work on interval index dropna
            cats = cats.dropna()
        data = pd.Categorical.from_codes(
            codes, categories=cats.to_pandas(), ordered=col.ordered
        )
        return pd.Index(data)

    def to_arrow(self) -> pa.Array:
        """Convert to PyArrow Array."""
        # pyarrow.Table doesn't support unsigned codes
        signed_type = (
            min_signed_type(self.codes.max())
            if self.size > 0
            else np.dtype(np.int8)
        )
        return pa.DictionaryArray.from_arrays(
            self.codes.astype(signed_type).to_arrow(),
            self.categories.to_arrow(),
            # TODO: Investigate if self.ordered can actually be None here
            ordered=self.ordered if self.ordered is not None else False,
        )

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> Self:
        return (
            self.astype(self.categories.dtype).clip(lo, hi).astype(self.dtype)  # type: ignore[return-value]
        )

    def where(
        self, cond: ColumnBase, other: ScalarLike | ColumnBase, inplace: bool
    ) -> ColumnBase:
        if is_scalar(other):
            try:
                other = self._encode(other)
            except ValueError:
                # When other is not present in categories,
                # fill with Null.
                other = None
            other = pa_scalar_to_plc_scalar(
                pa.scalar(
                    other,
                    type=cudf_dtype_to_pa_type(self.dtype._codes_dtype),
                )
            )
        elif isinstance(other.dtype, CategoricalDtype):
            other = other.codes  # type: ignore[union-attr]

        with access_columns(
            self.codes,
            other,
            cond,
            mode="read",
            scope="internal",
        ) as (codes, other, cond):
            other_plc = (
                other.plc_column if isinstance(other, ColumnBase) else other
            )
            result_plc = plc.copying.copy_if_else(
                codes.plc_column,
                other_plc,
                cond.plc_column,
            )
        return ColumnBase.create(result_plc, self.dtype)

    def _encode(self, value: ScalarLike) -> ScalarLike:
        return self.categories.find_first_value(value)

    def _decode(self, value: int) -> ScalarLike:
        if value == _DEFAULT_CATEGORICAL_VALUE:
            return None
        return self.categories.element_indexing(value)

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        """
        Return col with *to_replace* replaced with *replacement*.
        """
        to_replace_col = as_column(to_replace)
        if to_replace_col.is_all_null:
            to_replace_col = to_replace_col.astype(self.categories.dtype)
        replacement_col = as_column(replacement)
        if replacement_col.is_all_null:
            replacement_col = replacement_col.astype(self.categories.dtype)

        # TODO: This check looks incorrect, should we be checking the dtypes of the
        # columns instead of the column objects themselves?
        if type(to_replace_col) is not type(replacement_col):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )
        with to_replace_col.access(mode="read", scope="internal"):
            with replacement_col.access(mode="read", scope="internal"):
                distinct_table = plc.stream_compaction.stable_distinct(
                    plc.Table(
                        [
                            to_replace_col.plc_column,
                            replacement_col.plc_column,
                        ]
                    ),
                    [0],
                    plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                    plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL,
                )

        # Work directly with plc columns, only wrap when needed for high-level ops
        old_plc = distinct_table.columns()[0]
        new_plc = distinct_table.columns()[1]

        if old_plc.null_count() == 1:
            # Get the replacement value for the null in old_col
            old_isnull_plc = plc.unary.is_null(old_plc)
            filtered_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([new_plc]), old_isnull_plc
            )
            # We know there's exactly 1 null, so filtered result has 1 row
            fill_scalar = plc.copying.get_element(
                filtered_table.columns()[0], 0
            )
            fill_value = fill_scalar.to_arrow().as_py()

            # The `in` operator will only work on certain column types
            # (NumericalColumn, StringColumn).
            if fill_value in self.categories:  # type: ignore[operator]
                replaced = self.fillna(fill_value)
            else:
                new_categories = self.categories.append(
                    as_column([fill_value])
                )
                replaced = self._set_categories(new_categories)
                replaced = replaced.fillna(fill_value)

            # Drop rows where "old" column is null
            dropped = plc.stream_compaction.drop_nulls(
                plc.Table([old_plc, new_plc]), [0], 1
            )
            old_plc = dropped.columns()[0]
            new_plc = dropped.columns()[1]
        else:
            replaced = self

        if new_plc.null_count() > 0:
            # Get old values where new is null (these categories will be dropped)
            new_isnull_plc = plc.unary.is_null(new_plc)
            filtered_table = plc.stream_compaction.apply_boolean_mask(
                plc.Table([old_plc]), new_isnull_plc
            )
            drop_values = ColumnBase.create(
                filtered_table.columns()[0], to_replace_col.dtype
            )
            cur_categories = replaced.categories
            new_categories = cur_categories.apply_boolean_mask(
                cur_categories.isin(drop_values).unary_operator("not")  # type: ignore[arg-type]
            )
            replaced = replaced._set_categories(new_categories)

            # Drop rows where "new" column is null
            dropped = plc.stream_compaction.drop_nulls(
                plc.Table([old_plc, new_plc]), [1], 1
            )
            old_plc = dropped.columns()[0]
            new_plc = dropped.columns()[1]

        # Wrap for remaining operations that need ColumnBase
        old_col = ColumnBase.create(old_plc, to_replace_col.dtype)
        new_col = ColumnBase.create(new_plc, replacement_col.dtype)

        # create a dataframe containing the pre-replacement categories
        # and a column with the appropriate labels replaced.
        # The index of this dataframe represents the original
        # ints that map to the categories
        cats_col = as_column(replaced.dtype.categories)
        if old_col.dtype != cats_col.dtype and new_col.dtype != cats_col.dtype:
            cats_replace_col = cats_col.copy()
        else:
            with cats_col.access(mode="read", scope="internal"):
                cats_replace_plc = plc.replace.find_and_replace_all(
                    cats_col.plc_column,
                    old_col.plc_column,
                    new_col.plc_column,
                )
            cats_replace_col = ColumnBase.create(
                cats_replace_plc,
                cats_col.dtype,
            )

        # Construct the new categorical labels
        # If a category is being replaced by an existing one, we
        # want to map it to None. If it's totally new, we want to
        # map it to the new label it is to be replaced by.
        # plc.search.contains requires matching dtypes, so skip if types differ
        # (nothing would match anyway when types are different)
        if new_col.dtype == cats_col.dtype:
            with new_col.access(mode="read", scope="internal"):
                with cats_col.access(mode="read", scope="internal"):
                    # Check which replacement values are in categories
                    is_in_cats = plc.search.contains(
                        cats_col.plc_column,  # haystack
                        new_col.plc_column,  # needles
                    )
                    # Create a null scalar for replacement
                    null_scalar = plc.Scalar.from_py(
                        None,
                        new_col.plc_column.type(),
                    )
                    # Where is_in_cats is True, replace with null; otherwise keep original
                    dtype_replace_plc = plc.copying.copy_if_else(
                        null_scalar,  # value when True
                        new_col.plc_column,  # value when False
                        is_in_cats,
                    )
            dtype_replace_col = ColumnBase.create(
                dtype_replace_plc, new_col.dtype
            )
        else:
            # Types don't match, nothing in new_col can be in cats_col
            dtype_replace_col = new_col

        if (
            old_col.dtype != cats_col.dtype
            and dtype_replace_col.dtype != cats_col.dtype
        ):
            new_cats_col = cats_col.copy()
        else:
            with cats_col.access(mode="read", scope="internal"):
                with dtype_replace_col.access(mode="read", scope="internal"):
                    new_cats_plc = plc.replace.find_and_replace_all(
                        cats_col.plc_column,
                        old_col.plc_column,
                        dtype_replace_col.plc_column,
                    )
            new_cats_col = ColumnBase.create(new_cats_plc, cats_col.dtype)

        # Filter out categories that were mapped to None
        with new_cats_col.access(mode="read", scope="internal"):
            dropped = plc.stream_compaction.drop_nulls(
                plc.Table([new_cats_col.plc_column]), [0], 1
            )
        new_cats_col = ColumnBase.create(dropped.columns()[0], cats_col.dtype)
        new_index_col = as_column(range(len(new_cats_col)))

        # Join old categories with new categories to build a mapping
        # from old codes to new codes
        with cats_replace_col.access(mode="read", scope="internal"):
            with new_cats_col.access(mode="read", scope="internal"):
                left_keys = plc.Table([cats_replace_col.plc_column])
                right_keys = plc.Table([new_cats_col.plc_column])

                # Perform inner join - returns gather maps
                # Use NullEquality.UNEQUAL to match pandas behavior
                left_gather_map, right_gather_map = plc.join.inner_join(
                    left_keys,
                    right_keys,
                    plc.types.NullEquality.UNEQUAL,
                )

        # Build result by gathering from source columns
        # We need the index from left (old category positions) and "index" from right (new codes)
        # Left table is old_cats: columns are [cats_col, cats_replace_col]
        # The left gather map gives us which rows from old_cats matched
        # The row indices in old_cats ARE the old code values (0, 1, 2, ..., n-1)

        # Gather from new_index_col using right_gather_map to get new codes
        with new_index_col.access(mode="read", scope="internal"):
            right_table = plc.Table([new_index_col.plc_column])
            gathered_right = plc.copying.gather(
                right_table,
                right_gather_map,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
        gathered_new_index = ColumnBase.create(
            gathered_right.columns()[0], new_index_col.dtype
        )

        # The left_gather_map contains the old category indices that matched
        # These are the "to_replace" codes
        # left_gather_map is INT32 from pylibcudf, need to cast to codes dtype
        codes_to_replace = ColumnBase.create(
            left_gather_map, np.dtype("int32")
        ).astype(replaced.codes.dtype)
        codes_replacement = gathered_new_index.astype(replaced.codes.dtype)

        with replaced.codes.access(mode="read", scope="internal"):
            new_codes = plc.replace.find_and_replace_all(
                replaced.codes.plc_column,
                codes_to_replace.plc_column,
                codes_replacement.plc_column,
            )
        result = ColumnBase.create(
            new_codes,
            CategoricalDtype(
                categories=new_cats_col, ordered=self.dtype.ordered
            ),
        )
        if result.dtype != self.dtype:
            warnings.warn(
                "The behavior of replace with "
                "CategoricalDtype is deprecated. In a future version, replace "
                "will only be used for cases that preserve the categories. "
                "To change the categories, use ser.cat.rename_categories "
                "instead.",
                FutureWarning,
            )
        return result  # type: ignore[return-value]

    def isnull(self) -> ColumnBase:
        """
        Identify missing values in a CategoricalColumn.
        """
        result = super().isnull()

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = self.categories.isnan()
            if categories.any():
                code = self._encode(np.nan)
                result = result | (self.codes == code)

        return result

    def notnull(self) -> ColumnBase:
        """
        Identify non-missing values in a CategoricalColumn.
        """
        result = super().is_valid()

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = self.categories.isnan()
            if categories.any():
                code = self._encode(np.nan)
                result = result & (self.codes != code)

        return result

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if is_scalar(fill_value):
            if fill_value != _DEFAULT_CATEGORICAL_VALUE:
                try:
                    fill_value = self._encode(fill_value)
                except ValueError as err:
                    raise ValueError(
                        f"{fill_value=} must be in categories"
                    ) from err
            return pa_scalar_to_plc_scalar(
                pa.scalar(
                    fill_value,
                    type=cudf_dtype_to_pa_type(self.dtype._codes_dtype),
                )
            )
        else:
            fill_value = as_column(fill_value, nan_as_null=False)
            if isinstance(fill_value.dtype, CategoricalDtype):
                if self.dtype != fill_value.dtype:
                    raise TypeError(
                        "Cannot set a categorical with another without identical categories"
                    )
            else:
                raise TypeError(
                    "Cannot set a categorical with non-categorical data"
                )
            fill_value = cast("CategoricalColumn", fill_value)._set_categories(
                self.categories,
            )
            return fill_value.codes.astype(self.dtype._codes_dtype)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        return self.codes.indices_of(self._encode(value))

    @cached_property
    def is_monotonic_increasing(self) -> bool:
        return bool(self.ordered) and super().is_monotonic_increasing

    @cached_property
    def is_monotonic_decreasing(self) -> bool:
        return bool(self.ordered) and super().is_monotonic_decreasing

    def as_categorical_column(self, dtype: CategoricalDtype) -> Self:
        if not isinstance(self.categories, type(dtype.categories._column)):
            if isinstance(
                self.categories.dtype, cudf.StructDtype
            ) and isinstance(dtype.categories.dtype, cudf.IntervalDtype):
                return cast(
                    "Self",
                    ColumnBase.create(self.plc_column, dtype),
                )
            else:
                # Otherwise if both categories are of different Column types,
                # return a column full of nulls.
                codes_plc = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        None,
                        dtype_to_pylibcudf_type(dtype._codes_dtype),
                    ),
                    self.size,
                )
                return cast(
                    "Self",
                    ColumnBase.create(codes_plc, dtype),
                )

        return self.set_categories(
            new_categories=self.dtype.categories
            if dtype._categories is None
            else dtype.categories,
            ordered=bool(dtype.ordered),
        )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        return self._get_decategorized_column().as_numerical_column(dtype)

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        return self._get_decategorized_column().as_string_column(dtype)

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        return self._get_decategorized_column().as_datetime_column(dtype)

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return self._get_decategorized_column().as_timedelta_column(dtype)

    def _get_decategorized_column(self) -> ColumnBase:
        if self.is_all_null:
            # self.categories is empty; just return codes
            return self.codes
        gather_map = self.codes.astype(SIZE_TYPE_DTYPE).fillna(0)
        out = self.categories.take(gather_map)
        mask = self.mask
        new_null_count = self.null_count
        if self.offset > 0 and mask is not None:
            with mask.access(mode="read", scope="internal"):
                mask = cudf.core.buffer.as_buffer(
                    plc.null_mask.copy_bitmask_from_bitmask(
                        mask,
                        self.offset,
                        mask.size - self.offset,
                    )
                )
                new_null_count = plc.null_mask.null_count(
                    mask,
                    0,
                    mask.size,
                )
        out = out.set_mask(mask, new_null_count)
        return out

    def copy(self, deep: bool = True) -> Self:
        if deep:
            dtype_copy = CategoricalDtype(
                categories=self.categories.copy(),
                ordered=self.ordered,
            )
        else:
            dtype_copy = self.dtype
        plc_col = self.plc_column.copy() if deep else self.plc_column
        return cast("Self", ColumnBase.create(plc_col, dtype_copy))

    @cached_property
    def memory_usage(self) -> int:
        return self.categories.memory_usage + self.codes.memory_usage

    @staticmethod
    def _concat(
        objs: MutableSequence[CategoricalColumn],
    ) -> CategoricalColumn:
        # TODO: This function currently assumes it is being called from
        # column.concat_columns, at least to the extent that all the
        # preprocessing in that function has already been done. That should be
        # improved as the concatenation API is solidified.

        # Find the first non-null column:
        head = next(
            (obj for obj in objs if obj.null_count != len(obj)), objs[0]
        )

        # Combine and de-dupe the categories
        cats = concat_columns([o.categories for o in objs]).unique()
        objs = [o._set_categories(cats, is_unique=True) for o in objs]
        codes = [o.codes for o in objs]

        newsize = sum(map(len, codes))
        if newsize > np.iinfo(SIZE_TYPE_DTYPE).max:
            raise MemoryError(
                f"Result of concat cannot have size > {SIZE_TYPE_DTYPE}_MAX"
            )
        elif newsize == 0:
            codes_col = column_empty(0, head.codes.dtype)
        else:
            codes_col = concat_columns(codes)

        return cast(
            "Self",
            ColumnBase.create(
                codes_col.plc_column,
                CategoricalDtype(categories=cats),
            ),
        )

    def _with_type_metadata(self: Self, dtype: DtypeObj) -> Self:
        if isinstance(dtype, CategoricalDtype):
            return type(self)._from_preprocessed(
                plc_column=self.plc_column,
                dtype=dtype,
            )

        return self

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
    ) -> Self:
        # See CategoricalAccessor.set_categories.

        ordered = ordered if ordered is not None else self.ordered
        new_categories = as_column(new_categories)

        if isinstance(new_categories, CategoricalColumn):
            new_categories = new_categories.categories

        # when called with rename=True, the pandas behavior is
        # to replace the current category values with the new
        # categories.
        if rename:
            # enforce same length
            if len(new_categories) != len(self.categories):
                raise ValueError(
                    "new_categories must have the same "
                    "number of items as old categories"
                )
            return cast(
                "Self",
                ColumnBase.create(
                    self.plc_column,
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
                    ),
                ),
            )
        else:
            out_col = self
            if type(out_col.categories) is not type(new_categories):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                new_dtype = CategoricalDtype(
                    categories=new_categories, ordered=ordered
                )
                new_codes_plc = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        None,
                        dtype_to_pylibcudf_type(new_dtype._codes_dtype),
                    ),
                    self.size,
                )
                out_col = cast(
                    "Self",
                    ColumnBase.create(
                        new_codes_plc,
                        new_dtype,
                    ),
                )
            elif (
                not out_col._categories_equal(new_categories, ordered=True)
                or not self.ordered == ordered
            ):
                out_col = out_col._set_categories(
                    new_categories,
                    ordered=ordered,
                )
        return out_col

    def _categories_equal(
        self, new_categories: ColumnBase, ordered: bool = False
    ) -> bool:
        cur_categories = self.categories
        if len(new_categories) != len(cur_categories):
            return False
        if new_categories.dtype != cur_categories.dtype:
            return False
        # if order doesn't matter, sort before the equals call below
        if not ordered:
            cur_categories = _sort_column(cur_categories)
            new_categories = _sort_column(new_categories)
        return cur_categories.equals(new_categories)

    def _set_categories(
        self,
        new_categories: Any,
        is_unique: bool = False,
        ordered: bool = False,
    ) -> Self:
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*.

        Notes
        -----
        Assumes ``new_categories`` is the same dtype as the current categories
        """

        cur_cats = as_column(self.categories)
        new_cats = as_column(new_categories)

        # Join the old and new categories to build a map from
        # old to new codes, inserting na_sentinel for any old
        # categories that don't exist in the new categories

        # Ensure new_categories is unique first
        if not (is_unique or new_cats.is_unique):
            new_cats = new_cats.unique()

        if cur_cats.equals(new_cats, check_dtypes=True):
            # TODO: Internal usages don't always need a copy; add a copy keyword
            # as_ordered shallow copies
            return self.copy().as_ordered(ordered=ordered)

        # Keep original new_cats for the result, but may need casted version for join
        result_cats = new_cats

        # Cast to common dtype for the join (the original DataFrame merge did this)
        if cur_cats.dtype != new_cats.dtype:
            common_dtype = find_common_type([cur_cats.dtype, new_cats.dtype])
            cur_cats = cur_cats.astype(common_dtype)
            new_cats = new_cats.astype(common_dtype)

        cur_codes = self.codes
        out_code_dtype = min_unsigned_type(max(len(cur_cats), len(new_cats)))

        new_codes_col = as_column(range(len(new_cats)), dtype=out_code_dtype)

        # Left join to map old categories to new codes via category values
        with cur_cats.access(mode="read", scope="internal"):
            with new_cats.access(mode="read", scope="internal"):
                old_cats_key = plc.Table([cur_cats.plc_column])
                new_cats_key = plc.Table([new_cats.plc_column])

                # Left join to find matching categories
                # Use NullEquality.UNEQUAL to match pandas behavior (nulls don't match)
                left_map1, right_map1 = plc.join.left_join(
                    old_cats_key,
                    new_cats_key,
                    plc.types.NullEquality.UNEQUAL,
                )

        # Gather new_codes using right_map (may have nulls for non-matches)
        with new_codes_col.access(mode="read", scope="internal"):
            new_codes_table = plc.Table([new_codes_col.plc_column])
            joined_new_codes_table = plc.copying.gather(
                new_codes_table,
                right_map1,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            )

        # The join may have reordered results. We need to create a lookup table
        # where position i contains the new code for old category i.
        # Scatter the joined_new_codes back to positions indicated by left_map1
        # Create a null-initialized target table of size len(cur_cats)
        null_target = plc.Column.from_scalar(
            plc.Scalar.from_py(None, dtype_to_pylibcudf_type(out_code_dtype)),
            len(cur_cats),
        )
        # Scatter joined new codes to positions indicated by left_map1
        matched_new_codes_table = plc.copying.scatter(
            joined_new_codes_table,
            left_map1,
            plc.Table([null_target]),
        )
        # Now matched_new_codes_table[i] = new code for old category i (or null if not found)

        # Apply the mapping to actual data codes
        # For each code in cur_codes, look up the new code in matched_new_codes
        # This is just a gather operation since matched_new_codes is indexed by old code
        # BUT: cur_codes may have nulls (representing missing categorical values)
        # gather doesn't accept null indices, so we need to handle them specially
        with cur_codes.access(mode="read", scope="internal"):
            # Get validity mask for cur_codes
            cur_codes_valid_plc = plc.unary.is_valid(cur_codes.plc_column)
        cur_codes_valid = ColumnBase.create(
            cur_codes_valid_plc, np.dtype("bool")
        )
        with cur_codes.access(mode="read", scope="internal"):
            if cur_codes.null_count == 0:
                # No nulls, can gather directly
                final_new_codes_table = plc.copying.gather(
                    matched_new_codes_table,
                    cur_codes.plc_column,
                    plc.copying.OutOfBoundsPolicy.NULLIFY,
                )
            elif cur_codes.null_count == len(cur_codes):
                # All nulls, result is all nulls
                final_new_codes_table = plc.Table(
                    [
                        plc.Column.from_scalar(
                            plc.Scalar.from_py(
                                None, dtype_to_pylibcudf_type(out_code_dtype)
                            ),
                            len(cur_codes),
                        )
                    ]
                )
            else:
                # Mixed: fill nulls with 0 temporarily, gather, then restore nulls
                with cur_codes_valid.access(mode="read", scope="internal"):
                    # Replace nulls with 0 (a valid index)
                    zero_scalar = plc.Scalar.from_py(
                        0, cur_codes.plc_column.type()
                    )
                    filled_codes = plc.copying.copy_if_else(
                        cur_codes.plc_column,
                        zero_scalar,
                        cur_codes_valid.plc_column,
                    )
                    # Gather using filled codes
                    gathered_table = plc.copying.gather(
                        matched_new_codes_table,
                        filled_codes,
                        plc.copying.OutOfBoundsPolicy.NULLIFY,
                    )
                    # Restore nulls where cur_codes was null
                    null_scalar = plc.Scalar.from_py(
                        None, dtype_to_pylibcudf_type(out_code_dtype)
                    )
                    final_codes = plc.copying.copy_if_else(
                        gathered_table.columns()[0],
                        null_scalar,
                        cur_codes_valid.plc_column,
                    )
                final_new_codes_table = plc.Table([final_codes])

        ordered = ordered if ordered is not None else self.ordered
        new_codes_result = ColumnBase.create(
            final_new_codes_table.columns()[0], out_code_dtype
        )
        return cast(
            "Self",
            ColumnBase.create(
                new_codes_result.plc_column,
                CategoricalDtype(categories=result_cats, ordered=ordered),
            ),
        )

    def add_categories(self, new_categories: Any) -> Self:
        old_categories = self.categories
        new_categories = as_column(
            new_categories,
            dtype=old_categories.dtype if len(new_categories) == 0 else None,
        )
        if is_mixed_with_object_dtype(old_categories, new_categories):
            raise TypeError(
                f"cudf does not support adding categories with existing "
                f"categories of dtype `{old_categories.dtype}` and new "
                f"categories of dtype `{new_categories.dtype}`, please "
                f"type-cast new_categories to the same type as "
                f"existing categories."
            )
        common_dtype = find_common_type(
            [old_categories.dtype, new_categories.dtype]
        )

        new_categories = new_categories.astype(common_dtype)
        old_categories = old_categories.astype(common_dtype)

        if old_categories.isin(new_categories).any():
            raise ValueError("new categories must not include old categories")

        new_categories = old_categories.append(new_categories)
        if not self._categories_equal(new_categories):
            return self._set_categories(new_categories)
        return self

    def remove_categories(
        self,
        removals: Any,
    ) -> Self:
        removals = as_column(removals).astype(self.categories.dtype)
        removals_mask = removals.isin(self.categories)

        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            raise ValueError("removals must all be in old categories")

        new_categories = self.categories.apply_boolean_mask(
            self.categories.isin(removals).unary_operator("not")
        )
        if not self._categories_equal(new_categories):
            return self._set_categories(new_categories)
        return self

    def reorder_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
    ) -> CategoricalColumn:
        new_categories = as_column(new_categories)
        # Compare new_categories against current categories.
        # Ignore order for comparison because we're only interested
        # in whether new_categories has all the same values as the
        # current set of categories.
        if not self._categories_equal(new_categories, ordered=False):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        return self._set_categories(new_categories, ordered=ordered)

    def rename_categories(self, new_categories: Any) -> CategoricalColumn:
        raise NotImplementedError(
            "rename_categories is currently not supported."
        )

    def remove_unused_categories(self) -> Self:
        raise NotImplementedError(
            "remove_unused_categories is currently not supported."
        )

    def as_ordered(self, ordered: bool) -> Self:
        if self.dtype.ordered == ordered:
            return self
        return cast(
            "Self",
            ColumnBase.create(
                self.plc_column,
                CategoricalDtype(categories=self.categories, ordered=ordered),
            ),
        )
