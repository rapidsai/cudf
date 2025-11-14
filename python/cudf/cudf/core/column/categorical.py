# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Self

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core.column import column
from cudf.core.dtypes import CategoricalDtype, IntervalDtype
from cudf.utils.dtypes import (
    SIZE_TYPE_DTYPE,
    cudf_dtype_to_pa_type,
    find_common_type,
    is_mixed_with_object_dtype,
    min_signed_type,
    min_unsigned_type,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.utils import _is_null_host_scalar

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableSequence, Sequence

    from cudf._typing import (
        ColumnBinaryOperand,
        ColumnLike,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.buffer import Buffer
    from cudf.core.column import (
        ColumnBase,
        DatetimeColumn,
        NumericalColumn,
        StringColumn,
        TimeDeltaColumn,
    )


# Using np.int8(-1) to allow silent wrap-around when casting to uint
# it may make sense to make this dtype specific or a function.
_DEFAULT_CATEGORICAL_VALUE = np.int8(-1)


class CategoricalColumn(column.ColumnBase):
    """
    Implements operations for Columns of Categorical type

    Parameters
    ----------
    dtype : CategoricalDtype
    mask : Buffer
        The validity mask
    offset : int
        Data offset
    children : Tuple[ColumnBase]
        Two non-null columns containing the categories and codes
        respectively
    """

    dtype: CategoricalDtype
    _children: tuple[NumericalColumn]
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

    def __init__(
        self,
        plc_column: plc.Column,
        size: int,
        dtype: CategoricalDtype,
        offset: int,
        null_count: int,
        exposed: bool,
    ) -> None:
        if not isinstance(dtype, CategoricalDtype):
            raise ValueError(
                f"{dtype=} must be cudf.CategoricalDtype instance."
            )
        super().__init__(
            plc_column=plc_column,
            size=size,
            dtype=dtype,
            offset=offset,
            null_count=null_count,
            exposed=exposed,
        )
        self._codes = self.children[0].set_mask(self.mask)

    @classmethod
    def _get_data_buffer_from_pylibcudf_column(
        cls, plc_column: plc.Column, exposed: bool
    ) -> None:
        """
        This column considers the plc_column (i.e. codes) as children
        """
        return None

    def _get_children_from_pylibcudf_column(
        self, plc_column: plc.Column, dtype: DtypeObj, exposed: bool
    ) -> tuple[ColumnBase]:
        """
        This column considers the plc_column (i.e. codes) as children
        """
        return (
            type(self).from_pylibcudf(plc_column, data_ptr_exposed=exposed),
        )

    @property
    def base_size(self) -> int:
        return int(
            (self.base_children[0].size) / self.base_children[0].dtype.itemsize
        )

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            self._encode(item)
        except ValueError:
            return False
        return self._encode(item) in self.codes

    def _process_values_for_isin(
        self, values: Sequence
    ) -> tuple[ColumnBase, ColumnBase]:
        # Convert values to categorical dtype like self
        return self, column.as_column(values, dtype=self.dtype)

    def set_base_mask(self, value: Buffer | None) -> None:
        super().set_base_mask(value)
        self._codes = self.children[0].set_mask(self.mask)

    def set_base_children(self, value: tuple[NumericalColumn]) -> None:  # type: ignore[override]
        super().set_base_children(value)
        self._codes = value[0].set_mask(self.mask)

    @property
    def children(self) -> tuple[NumericalColumn]:
        if self.offset == 0 and self.size == self.base_size:
            return super().children  # type: ignore[return-value]
        if self._children is None:
            # Compute children from the column view (children factoring self.size)
            child = type(self).from_pylibcudf(
                self.to_pylibcudf(mode="read").copy()
            )
            self._children = (child,)
        return self._children

    @property
    def categories(self) -> ColumnBase:
        return self.dtype.categories._column

    @property
    def codes(self) -> NumericalColumn:
        return self._codes

    @property
    def ordered(self) -> bool | None:
        return self.dtype.ordered

    def __setitem__(self, key, value):
        if is_scalar(value) and _is_null_host_scalar(value):
            to_add_categories = 0
        else:
            if is_scalar(value):
                arr = column.as_column(value, length=1, nan_as_null=False)
            else:
                arr = column.as_column(value, nan_as_null=False)
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
            value = cudf.core.column.as_column(value).astype(self.dtype)
            value = value.codes
        codes = self.codes
        codes[key] = value
        out = codes._with_type_metadata(self.dtype)
        self._mimic_inplace(out, inplace=True)

    def _fill(
        self,
        fill_value: plc.Scalar,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        fill_code = self._encode(fill_value.to_arrow())
        result = self if inplace else self.copy()
        result.codes._fill(
            pa_scalar_to_plc_scalar(pa.scalar(fill_code)),
            begin,
            end,
            inplace=True,
        )
        return result

    def slice(self, start: int, stop: int, stride: int | None = None) -> Self:
        return self.codes.slice(start, stop, stride)._with_type_metadata(  # type: ignore[return-value]
            self.dtype
        )

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        *args,
        **kwargs,
    ) -> ScalarLike:
        # Only valid reductions are min and max
        if not self.ordered:
            raise TypeError(
                f"Categorical is not ordered for operation {op} "
                "you can use .as_ordered() to change the Categorical "
                "to an ordered one."
            )
        return self._decode(
            self.codes._reduce(op, skipna, min_count, *args, **kwargs)
        )

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        if isinstance(other, column.ColumnBase):
            if (
                isinstance(other, CategoricalColumn)
                and other.dtype != self.dtype
            ):
                raise TypeError(
                    "Categoricals can only compare with the same type"
                )
            # We'll compare self's decategorized values later for non-CategoricalColumn
        else:
            codes = column.as_column(
                self._encode(other), length=len(self), dtype=self.codes.dtype
            )
            other = codes._with_type_metadata(self.dtype)
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

    def sort_values(self, ascending: bool = True, na_position="last") -> Self:
        return self.codes.sort_values(  # type: ignore[return-value]
            ascending, na_position
        )._with_type_metadata(self.dtype)

    def element_indexing(self, index: int) -> ScalarLike:
        val = self.codes.element_indexing(index)
        if val is self._PANDAS_NA_VALUE:
            return val
        return self._decode(int(val))  # type: ignore[arg-type]

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
            new_mask = self.notnull().fillna(False).as_mask()
            col = self.set_mask(new_mask)
        else:
            col = self

        signed_dtype = min_signed_type(len(col.categories))
        codes = (
            col.codes.astype(signed_dtype)
            .fillna(_DEFAULT_CATEGORICAL_VALUE)
            .values_host
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
            if self.codes.size > 0
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

    def unique(self) -> Self:
        return self.codes.unique()._with_type_metadata(self.dtype)  # type: ignore[return-value]

    def _cast_self_and_other_for_where(
        self, other: ScalarLike | ColumnBase, inplace: bool
    ) -> tuple[ColumnBase, plc.Scalar | ColumnBase]:
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
                    type=cudf_dtype_to_pa_type(self.codes.dtype),
                )
            )
        elif isinstance(other.dtype, CategoricalDtype):
            other = other.codes  # type: ignore[union-attr]

        return self.codes, other

    def _encode(self, value) -> ScalarLike:
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
        to_replace_col = column.as_column(to_replace)
        if len(to_replace_col) == to_replace_col.null_count:
            to_replace_col = to_replace_col.astype(self.categories.dtype)
        replacement_col = column.as_column(replacement)
        if len(replacement_col) == replacement_col.null_count:
            replacement_col = replacement_col.astype(self.categories.dtype)

        if type(to_replace_col) is not type(replacement_col):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )
        df = cudf.DataFrame._from_data(
            {"old": to_replace_col, "new": replacement_col}
        )
        df = df.drop_duplicates(subset=["old"], keep="last", ignore_index=True)
        if df._data["old"].null_count == 1:
            fill_value = (
                df._data["new"]
                .apply_boolean_mask(df._data["old"].isnull())
                .element_indexing(0)
            )
            # TODO: This line of code does not work because we cannot use the
            # `in` operator on self.categories (which is a column). mypy
            # realizes that this is wrong because __iter__ is not implemented.
            # However, it seems that this functionality has been broken for a
            # long time so for now we're just having mypy ignore and we'll come
            # back to this.
            if fill_value in self.categories:  # type: ignore[operator]
                replaced = self.fillna(fill_value)
            else:
                new_categories = self.categories.append(
                    column.as_column([fill_value])
                )
                replaced = self._set_categories(new_categories)
                replaced = replaced.fillna(fill_value)
            df = df.dropna(subset=["old"])
            to_replace_col = df._data["old"]
            replacement_col = df._data["new"]
        else:
            replaced = self
        if df._data["new"].null_count > 0:
            drop_values = df._data["old"].apply_boolean_mask(
                df._data["new"].isnull()
            )
            cur_categories = replaced.categories
            new_categories = cur_categories.apply_boolean_mask(
                cur_categories.isin(drop_values).unary_operator("not")
            )
            replaced = replaced._set_categories(new_categories)
            df = df.dropna(subset=["new"])
            to_replace_col = df._data["old"]
            replacement_col = df._data["new"]

        # create a dataframe containing the pre-replacement categories
        # and a column with the appropriate labels replaced.
        # The index of this dataframe represents the original
        # ints that map to the categories
        cats_col = column.as_column(replaced.dtype.categories)
        old_cats = cudf.DataFrame._from_data(
            {
                "cats": cats_col,
                "cats_replace": cats_col.find_and_replace(
                    to_replace_col, replacement_col
                ),
            }
        )

        # Construct the new categorical labels
        # If a category is being replaced by an existing one, we
        # want to map it to None. If it's totally new, we want to
        # map it to the new label it is to be replaced by
        dtype_replace = cudf.Series._from_column(replacement_col)
        dtype_replace[dtype_replace.isin(cats_col)] = None
        new_cats_col = cats_col.find_and_replace(
            to_replace_col, dtype_replace._column
        )

        # anything we mapped to None, we want to now filter out since
        # those categories don't exist anymore
        # Resetting the index creates a column 'index' that associates
        # the original integers to the new labels
        bmask = new_cats_col.notnull()
        new_cats_col = new_cats_col.apply_boolean_mask(bmask)
        new_cats = cudf.DataFrame._from_data(
            {
                "index": column.as_column(range(len(new_cats_col))),
                "cats": new_cats_col,
            }
        )

        # old_cats contains replaced categories and the ints that
        # previously mapped to those categories and the index of
        # new_cats is a RangeIndex that contains the new ints
        catmap = old_cats.merge(
            new_cats, left_on="cats_replace", right_on="cats", how="inner"
        )

        # The index of this frame is now the old ints, but the column
        # named 'index', which came from the filtered categories,
        # contains the new ints that we need to map to
        to_replace_col = column.as_column(catmap.index).astype(
            replaced.codes.dtype
        )
        replacement_col = catmap._data["index"].astype(replaced.codes.dtype)

        replaced_codes = column.as_column(replaced.codes)
        new_codes = replaced_codes.replace(to_replace_col, replacement_col)
        result = new_codes._with_type_metadata(
            CategoricalDtype(
                categories=new_cats["cats"], ordered=self.dtype.ordered
            )
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
                    fill_value, type=cudf_dtype_to_pa_type(self.codes.dtype)
                )
            )
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
            if isinstance(fill_value.dtype, CategoricalDtype):
                if self.dtype != fill_value.dtype:
                    raise TypeError(
                        "Cannot set a categorical with another without identical categories"
                    )
            else:
                raise TypeError(
                    "Cannot set a categorical with non-categorical data"
                )
            fill_value = cast(CategoricalColumn, fill_value)._set_categories(
                self.categories,
            )
            return fill_value.codes.astype(self.codes.dtype)

    def indices_of(self, value: ScalarLike) -> NumericalColumn:
        return self.codes.indices_of(self._encode(value))

    @property
    def is_monotonic_increasing(self) -> bool:
        return bool(self.ordered) and self.codes.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        return bool(self.ordered) and self.codes.is_monotonic_decreasing

    def as_categorical_column(self, dtype: CategoricalDtype) -> Self:
        if not isinstance(self.categories, type(dtype.categories._column)):
            if isinstance(
                self.categories.dtype, cudf.StructDtype
            ) and isinstance(dtype.categories.dtype, cudf.IntervalDtype):
                return self._with_type_metadata(dtype)
            else:
                # Otherwise if both categories are of different Column types,
                # return a column full of nulls.
                codes = cast(
                    cudf.core.column.numerical.NumericalColumn,
                    column.as_column(
                        _DEFAULT_CATEGORICAL_VALUE,
                        length=self.size,
                        dtype=self.codes.dtype,
                    ),
                )
                return codes._with_type_metadata(dtype)  # type: ignore[return-value]

        return self.set_categories(
            new_categories=dtype.categories, ordered=bool(dtype.ordered)
        )

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        return self._get_decategorized_column().as_numerical_column(dtype)

    def as_string_column(self, dtype) -> StringColumn:
        return self._get_decategorized_column().as_string_column(dtype)

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        return self._get_decategorized_column().as_datetime_column(dtype)

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        return self._get_decategorized_column().as_timedelta_column(dtype)

    def _get_decategorized_column(self) -> ColumnBase:
        if self.null_count == len(self):
            # self.categories is empty; just return codes
            return self.codes
        gather_map = self.codes.astype(SIZE_TYPE_DTYPE).fillna(0)
        out = self.categories.take(gather_map)
        out = out.set_mask(self.mask)
        return out

    def copy(self, deep: bool = True) -> Self:
        result_col = super().copy(deep=deep)
        if deep:
            dtype_copy = CategoricalDtype(
                categories=self.categories.copy(),
                ordered=self.ordered,
            )
            result_col = cast(Self, result_col._with_type_metadata(dtype_copy))
        return result_col

    @cached_property
    def memory_usage(self) -> int:
        return self.categories.memory_usage + self.codes.memory_usage

    def _mimic_inplace(
        self, other_col: ColumnBase, inplace: bool = False
    ) -> Self | None:
        out = super()._mimic_inplace(other_col, inplace=inplace)  # type: ignore[arg-type]
        if inplace and isinstance(other_col, CategoricalColumn):
            self._codes = other_col.codes
        return out

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
        cats = column.concat_columns([o.categories for o in objs]).unique()
        objs = [o._set_categories(cats, is_unique=True) for o in objs]
        codes = [o.codes for o in objs]

        newsize = sum(map(len, codes))
        if newsize > np.iinfo(SIZE_TYPE_DTYPE).max:
            raise MemoryError(
                f"Result of concat cannot have size > {SIZE_TYPE_DTYPE}_MAX"
            )
        elif newsize == 0:
            codes_col = column.column_empty(0, head.codes.dtype)
        else:
            codes_col = column.concat_columns(codes)

        return codes_col._with_type_metadata(CategoricalDtype(categories=cats))  # type: ignore[return-value]

    def _with_type_metadata(self: Self, dtype: DtypeObj) -> Self:
        if isinstance(dtype, CategoricalDtype):
            return type(self)(
                plc_column=self.plc_column,
                size=self.size,
                dtype=dtype,
                offset=self.offset,
                null_count=self.null_count,
                exposed=False,
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
        new_categories = column.as_column(new_categories)

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
            return self._with_type_metadata(
                CategoricalDtype(categories=new_categories, ordered=ordered)
            )
        else:
            out_col = self
            if type(out_col.categories) is not type(new_categories):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                new_codes = cast(
                    cudf.core.column.numerical.NumericalColumn,
                    column.as_column(
                        _DEFAULT_CATEGORICAL_VALUE,
                        length=self.size,
                        dtype=self.codes.dtype,
                    ),
                )
                out_col = new_codes._with_type_metadata(  # type: ignore[assignment]
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
                    )
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
        self, new_categories: ColumnBase, ordered=False
    ) -> bool:
        cur_categories = self.categories
        if len(new_categories) != len(cur_categories):
            return False
        if new_categories.dtype != cur_categories.dtype:
            return False
        # if order doesn't matter, sort before the equals call below
        if not ordered:
            cur_categories = cur_categories.sort_values()
            new_categories = new_categories.sort_values()
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

        cur_cats = column.as_column(self.categories)
        new_cats = column.as_column(new_categories)

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

        cur_codes = self.codes
        out_code_dtype = min_unsigned_type(max(len(cur_cats), len(new_cats)))

        cur_order = column.as_column(range(len(cur_codes)))
        old_codes = column.as_column(
            range(len(cur_cats)), dtype=out_code_dtype
        )
        new_codes = column.as_column(
            range(len(new_cats)), dtype=out_code_dtype
        )

        new_df = cudf.DataFrame._from_data(
            data={"new_codes": new_codes, "cats": new_cats}
        )
        old_df = cudf.DataFrame._from_data(
            data={"old_codes": old_codes, "cats": cur_cats}
        )
        cur_df = cudf.DataFrame._from_data(
            data={"old_codes": cur_codes, "order": cur_order}
        )

        # Join the old and new categories and line up their codes
        df = old_df.merge(new_df, on="cats", how="left")
        # Join the old and new codes to "recode" the codes data buffer
        df = cur_df.merge(df, on="old_codes", how="left")
        df = df.sort_values(by="order")
        df.reset_index(drop=True, inplace=True)

        ordered = ordered if ordered is not None else self.ordered
        new_codes = cast(
            cudf.core.column.numerical.NumericalColumn, df._data["new_codes"]
        )
        return new_codes._with_type_metadata(  # type: ignore[return-value]
            CategoricalDtype(categories=new_cats, ordered=ordered)
        )

    def add_categories(self, new_categories: Any) -> Self:
        old_categories = self.categories
        new_categories = column.as_column(
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
        removals = column.as_column(removals).astype(self.categories.dtype)
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
        new_categories = column.as_column(new_categories)
        # Compare new_categories against current categories.
        # Ignore order for comparison because we're only interested
        # in whether new_categories has all the same values as the
        # current set of categories.
        if not self._categories_equal(new_categories, ordered=False):
            raise ValueError(
                "items in new_categories are not the same as in old categories"
            )
        return self._set_categories(new_categories, ordered=ordered)

    def rename_categories(self, new_categories) -> CategoricalColumn:
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
        return self._with_type_metadata(
            CategoricalDtype(categories=self.categories, ordered=ordered)
        )
