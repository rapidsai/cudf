# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import warnings
from collections import abc
from functools import cached_property
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda
from typing_extensions import Self

import cudf
from cudf import _lib as libcudf
from cudf._lib.transform import bools_to_mask
from cudf._typing import ColumnBinaryOperand, ColumnLike, Dtype, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.core.column.methods import ColumnMethods
from cudf.core.dtypes import CategoricalDtype, IntervalDtype
from cudf.utils.dtypes import (
    find_common_type,
    is_mixed_with_object_dtype,
    min_signed_type,
    min_unsigned_type,
)

if TYPE_CHECKING:
    from cudf._typing import SeriesOrIndex, SeriesOrSingleColumnIndex
    from cudf.core.column import (
        ColumnBase,
        DatetimeColumn,
        NumericalColumn,
        StringColumn,
        TimeDeltaColumn,
    )


_DEFAULT_CATEGORICAL_VALUE = -1


class CategoricalAccessor(ColumnMethods):
    """
    Accessor object for categorical properties of the Series values.
    Be aware that assigning to `categories` is a inplace operation,
    while all methods return new categorical data per default.

    Parameters
    ----------
    column : Column
    parent : Series or CategoricalIndex

    Examples
    --------
    >>> s = cudf.Series([1,2,3], dtype='category')
    >>> s
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1, 2, 3]
    >>> s.cat.categories
    Index([1, 2, 3], dtype='int64')
    >>> s.cat.reorder_categories([3,2,1])
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [3, 2, 1]
    >>> s.cat.remove_categories([1])
    0    <NA>
    1       2
    2       3
    dtype: category
    Categories (2, int64): [2, 3]
    >>> s.cat.set_categories(list('abcde'))
    0    <NA>
    1    <NA>
    2    <NA>
    dtype: category
    Categories (5, object): ['a', 'b', 'c', 'd', 'e']
    >>> s.cat.as_ordered()
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1 < 2 < 3]
    >>> s.cat.as_unordered()
    0    1
    1    2
    2    3
    dtype: category
    Categories (3, int64): [1, 2, 3]
    """

    _column: CategoricalColumn

    def __init__(self, parent: SeriesOrSingleColumnIndex):
        if not isinstance(parent.dtype, CategoricalDtype):
            raise AttributeError(
                "Can only use .cat accessor with a 'category' dtype"
            )
        super().__init__(parent=parent)

    @property
    def categories(self) -> "cudf.core.index.Index":
        """
        The categories of this categorical.
        """
        return self._column.dtype.categories

    @property
    def codes(self) -> "cudf.Series":
        """
        Return Series of codes as well as the index.
        """
        index = (
            self._parent.index
            if isinstance(self._parent, cudf.Series)
            else None
        )
        return cudf.Series(self._column.codes, index=index)

    @property
    def ordered(self) -> bool:
        """
        Whether the categories have an ordered relationship.
        """
        return self._column.ordered

    def as_ordered(self) -> Optional[SeriesOrIndex]:
        """
        Set the Categorical to be ordered.

        Returns
        -------
        Categorical
            Ordered Categorical.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.as_ordered()
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1 < 2 < 10]
        """
        return self._return_or_inplace(self._column.as_ordered(ordered=True))

    def as_unordered(self) -> Optional[SeriesOrIndex]:
        """
        Set the Categorical to be unordered.

        Returns
        -------
        Categorical
            Unordered Categorical or None if inplace.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s = s.cat.as_ordered()
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1 < 2 < 10]
        >>> s.cat.as_unordered()
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        """
        return self._return_or_inplace(self._column.as_ordered(ordered=False))

    def add_categories(self, new_categories: Any) -> Optional[SeriesOrIndex]:
        """
        Add new categories.

        `new_categories` will be included at the last/highest
        place in the categories and will be unused directly
        after this call.

        Parameters
        ----------
        new_categories : category or list-like of category
            The new categories to be included.

        Returns
        -------
        cat
            Categorical with new categories added.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 2], dtype="category")
        >>> s
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]
        >>> s.cat.add_categories([0, 3, 4])
        0    1
        1    2
        dtype: category
        Categories (5, int64): [1, 2, 0, 3, 4]
        >>> s
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]
        """
        old_categories = self._column.categories
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
        out_col = self._column
        if not out_col._categories_equal(new_categories):
            out_col = out_col._set_categories(new_categories)

        return self._return_or_inplace(out_col)

    def remove_categories(
        self,
        removals: Any,
    ) -> Optional[SeriesOrIndex]:
        """
        Remove the specified categories.

        `removals` must be included in the
        old categories. Values which were in the
        removed categories will be set to null.

        Parameters
        ----------
        removals : category or list-like of category
            The categories which should be removed.

        Returns
        -------
        cat
            Categorical with removed categories

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.remove_categories([1])
        0      10
        1    <NA>
        2    <NA>
        3       2
        4      10
        5       2
        6      10
        dtype: category
        Categories (2, int64): [2, 10]
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        """

        cats = self.categories.to_series()
        removals = cudf.Series(removals, dtype=cats.dtype)
        removals_mask = removals.isin(cats)

        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            vals = removals[~removals_mask].to_numpy()
            raise ValueError(f"removals must all be in old categories: {vals}")

        new_categories = cats[~cats.isin(removals)]._column
        out_col = self._column
        if not out_col._categories_equal(new_categories):
            out_col = out_col._set_categories(new_categories)

        return self._return_or_inplace(out_col)

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
    ) -> Optional[SeriesOrIndex]:
        """
        Set the categories to the specified new_categories.


        `new_categories` can include new categories (which
        will result in unused categories) or remove old categories
        (which results in values set to null). If `rename==True`,
        the categories will simple be renamed (less or more items
        than in old categories will result in values set to null or
        in unused categories respectively).

        This method can be used to perform more than one action
        of adding, removing, and reordering simultaneously and
        is therefore faster than performing the individual steps
        via the more specialised methods.

        On the other hand this methods does not do checks
        (e.g., whether the old categories are included in the
        new categories on a reorder), which can result in
        surprising changes.

        Parameters
        ----------
        new_categories : list-like
            The categories in new order.
        ordered : bool, default None
            Whether or not the categorical is treated as
            a ordered categorical. If not given, do
            not change the ordered information.
        rename : bool, default False
            Whether or not the `new_categories` should be
            considered as a rename of the old categories
            or as reordered categories.

        Returns
        -------
        cat
            Categorical with reordered categories

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([1, 1, 2, 10, 2, 10], dtype='category')
        >>> s
        0     1
        1     1
        2     2
        3    10
        4     2
        5    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.set_categories([1, 10])
        0       1
        1       1
        2    <NA>
        3      10
        4    <NA>
        5      10
        dtype: category
        Categories (2, int64): [1, 10]
        """
        return self._return_or_inplace(
            self._column.set_categories(
                new_categories=new_categories, ordered=ordered, rename=rename
            )
        )

    def reorder_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
    ) -> Optional[SeriesOrIndex]:
        """
        Reorder categories as specified in new_categories.

        `new_categories` need to include all old categories
        and no new category items.

        Parameters
        ----------
        new_categories : Index-like
            The categories in new order.
        ordered : bool, optional
            Whether or not the categorical is treated
            as a ordered categorical. If not given, do
            not change the ordered information.

        Returns
        -------
        cat
            Categorical with reordered categories

        Raises
        ------
        ValueError
            If the new categories do not contain all old
            category items or any new ones.


        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 1, 1, 2, 10, 2, 10], dtype="category")
        >>> s
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [1, 2, 10]
        >>> s.cat.reorder_categories([10, 1, 2])
        0    10
        1     1
        2     1
        3     2
        4    10
        5     2
        6    10
        dtype: category
        Categories (3, int64): [10, 1, 2]
        >>> s.cat.reorder_categories([10, 1])
        ValueError: items in new_categories are not the same as in
        old categories
        """
        return self._return_or_inplace(
            self._column.reorder_categories(new_categories, ordered=ordered),
        )


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

    dtype: cudf.core.dtypes.CategoricalDtype
    _codes: Optional[NumericalColumn]
    _children: Tuple[NumericalColumn]
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

    def __init__(
        self,
        dtype: CategoricalDtype,
        mask: Optional[Buffer] = None,
        size: Optional[int] = None,
        offset: int = 0,
        null_count: Optional[int] = None,
        children: Tuple["column.ColumnBase", ...] = (),
    ):
        if size is None:
            for child in children:
                assert child.offset == 0
                assert child.base_mask is None
            size = children[0].size
            size = size - offset
        if isinstance(dtype, pd.api.types.CategoricalDtype):
            dtype = CategoricalDtype.from_pandas(dtype)
        if not isinstance(dtype, CategoricalDtype):
            raise ValueError("dtype must be instance of CategoricalDtype")
        super().__init__(
            data=None,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )
        self._codes = None

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

    def set_base_data(self, value):
        if value is not None:
            raise RuntimeError(
                "CategoricalColumns do not use data attribute of Column, use "
                "`set_base_children` instead"
            )
        else:
            super().set_base_data(value)

    def _process_values_for_isin(
        self, values: Sequence
    ) -> Tuple[ColumnBase, ColumnBase]:
        lhs = self
        # We need to convert values to same type as self,
        # hence passing dtype=self.dtype
        rhs = cudf.core.column.as_column(values, dtype=self.dtype)
        return lhs, rhs

    def set_base_mask(self, value: Optional[Buffer]):
        super().set_base_mask(value)
        self._codes = None

    def set_base_children(self, value: Tuple[ColumnBase, ...]):
        super().set_base_children(value)
        self._codes = None

    @property
    def children(self) -> Tuple[NumericalColumn]:
        if self._children is None:
            codes_column = self.base_children[0]
            start = self.offset * codes_column.dtype.itemsize
            end = start + self.size * codes_column.dtype.itemsize
            codes_column = cast(
                cudf.core.column.NumericalColumn,
                column.build_column(
                    data=codes_column.base_data[start:end],
                    dtype=codes_column.dtype,
                    size=self.size,
                ),
            )
            self._children = (codes_column,)
        return self._children

    @property
    def categories(self) -> ColumnBase:
        return self.dtype.categories._values

    @property
    def codes(self) -> NumericalColumn:
        if self._codes is None:
            self._codes = self.children[0].set_mask(self.mask)
        return cast(cudf.core.column.NumericalColumn, self._codes)

    @property
    def ordered(self) -> bool:
        return self.dtype.ordered

    def __setitem__(self, key, value):
        if cudf.api.types.is_scalar(
            value
        ) and cudf._lib.scalar._is_null_host_scalar(value):
            to_add_categories = 0
        else:
            if cudf.api.types.is_scalar(value):
                arr = [value]
            else:
                arr = value
            to_add_categories = len(
                cudf.Index(arr, nan_as_null=False).difference(self.categories)
            )

        if to_add_categories > 0:
            raise TypeError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        if cudf.api.types.is_scalar(value):
            value = self._encode(value) if value is not None else value
        else:
            value = cudf.core.column.as_column(value).astype(self.dtype)
            value = value.codes
        codes = self.codes
        codes[key] = value
        out = cudf.core.column.build_categorical_column(
            categories=self.categories,
            codes=codes,
            mask=codes.base_mask,
            size=codes.size,
            offset=self.offset,
            ordered=self.ordered,
        )
        self._mimic_inplace(out, inplace=True)

    def _fill(
        self,
        fill_value: ScalarLike,
        begin: int,
        end: int,
        inplace: bool = False,
    ) -> Self:
        if end <= begin or begin >= self.size:
            return self if inplace else self.copy()

        fill_code = self._encode(fill_value)
        fill_scalar = cudf._lib.scalar.as_device_scalar(
            fill_code, self.codes.dtype
        )

        result = self if inplace else self.copy()

        libcudf.filling.fill_in_place(result.codes, begin, end, fill_scalar)
        return result

    def slice(
        self, start: int, stop: int, stride: Optional[int] = None
    ) -> Self:
        codes = self.codes.slice(start, stop, stride)
        return cast(
            Self,
            cudf.core.column.build_categorical_column(
                categories=self.categories,
                codes=cudf.core.column.build_column(
                    codes.base_data, dtype=codes.dtype
                ),
                mask=codes.base_mask,
                ordered=self.ordered,
                size=codes.size,
                offset=codes.offset,
            ),
        )

    def _reduce(
        self,
        op: str,
        skipna: Optional[bool] = None,
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
        other = self._wrap_binop_normalization(other)
        # TODO: This is currently just here to make mypy happy, but eventually
        # we'll need to properly establish the APIs for these methods.
        if not isinstance(other, CategoricalColumn):
            raise ValueError
        # Note: at this stage we are guaranteed that the dtypes are equal.
        if not self.ordered and op not in {
            "__eq__",
            "__ne__",
            "NULL_EQUALS",
            "NULL_NOT_EQUALS",
        }:
            raise TypeError(
                "The only binary operations supported by unordered "
                "categorical columns are equality and inequality."
            )
        return self.codes._binaryop(other.codes, op)

    def normalize_binop_value(self, other: ScalarLike) -> CategoricalColumn:
        if isinstance(other, column.ColumnBase):
            if not isinstance(other, CategoricalColumn):
                return NotImplemented
            if other.dtype != self.dtype:
                raise TypeError(
                    "Categoricals can only compare with the same type"
                )
            return other

        ary = column.as_column(
            self._encode(other), length=len(self), dtype=self.codes.dtype
        )
        return column.build_categorical_column(
            categories=self.dtype.categories._values,
            codes=column.as_column(ary),
            mask=self.base_mask,
            ordered=self.dtype.ordered,
        )

    def sort_values(
        self, ascending: bool = True, na_position="last"
    ) -> CategoricalColumn:
        codes = self.codes.sort_values(ascending, na_position)
        col = column.build_categorical_column(
            categories=self.dtype.categories._values,
            codes=column.build_column(codes.base_data, dtype=codes.dtype),
            mask=codes.base_mask,
            size=codes.size,
            ordered=self.dtype.ordered,
        )
        return col

    def element_indexing(self, index: int) -> ScalarLike:
        val = self.codes.element_indexing(index)
        return self._decode(int(val)) if val is not None else val

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
        index: Optional[pd.Index] = None,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Series:
        if nullable:
            raise NotImplementedError(f"{nullable=} is not implemented.")
        elif arrow_type:
            raise NotImplementedError(f"{arrow_type=} is not implemented.")

        if self.categories.dtype.kind == "f":
            new_mask = bools_to_mask(self.notnull())
            col = column.build_categorical_column(
                categories=self.categories,
                codes=column.as_column(self.codes, dtype=self.codes.dtype),
                mask=new_mask,
                ordered=self.dtype.ordered,
                size=self.codes.size,
            )
        else:
            col = self

        signed_dtype = min_signed_type(len(col.categories))
        codes = (
            col.codes.astype(signed_dtype)
            .fillna(_DEFAULT_CATEGORICAL_VALUE)
            .values_host
        )

        cats = col.categories
        if cats.dtype.kind in "biuf":
            cats = cats.nans_to_nulls().dropna()  # type: ignore[attr-defined]
        elif not isinstance(cats.dtype, IntervalDtype):
            # leaving out dropna because it temporarily changes an interval
            # index into a struct and throws off results.
            # TODO: work on interval index dropna
            cats = cats.dropna()
        data = pd.Categorical.from_codes(
            codes, categories=cats.to_pandas(), ordered=col.ordered
        )
        return pd.Series(data, index=index)

    def to_arrow(self) -> pa.Array:
        """Convert to PyArrow Array."""
        # arrow doesn't support unsigned codes
        signed_type = (
            min_signed_type(self.codes.max())
            if self.codes.size > 0
            else np.int8
        )
        codes = self.codes.astype(signed_type)
        categories = self.categories

        out_indices = codes.to_arrow()
        out_dictionary = categories.to_arrow()

        return pa.DictionaryArray.from_arrays(
            out_indices,
            out_dictionary,
            ordered=self.ordered,
        )

    @property
    def values_host(self) -> np.ndarray:
        """
        Return a numpy representation of the CategoricalColumn.
        """
        return self.to_pandas().values

    @property
    def values(self):
        """
        Return a CuPy representation of the CategoricalColumn.
        """
        raise NotImplementedError("cudf.Categorical is not yet implemented")

    def clip(self, lo: ScalarLike, hi: ScalarLike) -> "column.ColumnBase":
        return (
            self.astype(self.categories.dtype).clip(lo, hi).astype(self.dtype)
        )

    def data_array_view(
        self, *, mode="write"
    ) -> cuda.devicearray.DeviceNDArray:
        return self.codes.data_array_view(mode=mode)

    def unique(self) -> CategoricalColumn:
        codes = self.codes.unique()
        return column.build_categorical_column(
            categories=self.categories,
            codes=column.build_column(codes.base_data, dtype=codes.dtype),
            mask=codes.base_mask,
            offset=codes.offset,
            size=codes.size,
            ordered=self.ordered,
        )

    def _encode(self, value) -> ScalarLike:
        return self.categories.find_first_value(value)

    def _decode(self, value: int) -> ScalarLike:
        if value == _DEFAULT_CATEGORICAL_VALUE:
            return None
        return self.categories.element_indexing(value)

    def find_and_replace(
        self,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> CategoricalColumn:
        """
        Return col with *to_replace* replaced with *replacement*.
        """
        to_replace_col = column.as_column(to_replace)
        if len(to_replace_col) == to_replace_col.null_count:
            to_replace_col = to_replace_col.astype(self.categories.dtype)
        replacement_col = column.as_column(replacement)
        if len(replacement_col) == replacement_col.null_count:
            replacement_col = replacement_col.astype(self.categories.dtype)

        if type(to_replace_col) != type(replacement_col):
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
            if fill_value in self.categories:  # type: ignore
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
                ~cudf.Series(cur_categories.isin(drop_values))
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
        dtype_replace = cudf.Series._from_data({None: replacement_col})
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

        replaced = column.as_column(replaced.codes)
        output = libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
        )

        result = column.build_categorical_column(
            categories=new_cats["cats"],
            codes=column.build_column(output.base_data, dtype=output.dtype),
            mask=output.base_mask,
            offset=output.offset,
            size=output.size,
            ordered=self.dtype.ordered,
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
        return result

    def isnull(self) -> ColumnBase:
        """
        Identify missing values in a CategoricalColumn.
        """
        result = libcudf.unary.is_null(self)

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = libcudf.unary.is_nan(self.categories)
            if categories.any():
                code = self._encode(np.nan)
                result = result | (self.codes == cudf.Scalar(code))

        return result

    def notnull(self) -> ColumnBase:
        """
        Identify non-missing values in a CategoricalColumn.
        """
        result = libcudf.unary.is_valid(self)

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values in case
            # of an underlying float column
            categories = libcudf.unary.is_nan(self.categories)
            if categories.any():
                code = self._encode(np.nan)
                result = result & (self.codes != cudf.Scalar(code))

        return result

    def fillna(
        self,
        fill_value: Any = None,
        method: Optional[str] = None,
    ) -> Self:
        """
        Fill null values with *fill_value*
        """
        if fill_value is not None:
            fill_is_scalar = np.isscalar(fill_value)

            if fill_is_scalar:
                if fill_value == _DEFAULT_CATEGORICAL_VALUE:
                    fill_value = self.codes.dtype.type(fill_value)
                else:
                    try:
                        fill_value = self._encode(fill_value)
                        fill_value = self.codes.dtype.type(fill_value)
                    except ValueError as err:
                        err_msg = "fill value must be in categories"
                        raise ValueError(err_msg) from err
            else:
                fill_value = column.as_column(fill_value, nan_as_null=False)
                if isinstance(fill_value, CategoricalColumn):
                    if self.dtype != fill_value.dtype:
                        raise TypeError(
                            "Cannot set a Categorical with another, "
                            "without identical categories"
                        )
                # TODO: only required if fill_value has a subset of the
                # categories:
                fill_value = fill_value._set_categories(
                    self.categories,
                    is_unique=True,
                )
                fill_value = column.as_column(fill_value.codes).astype(
                    self.codes.dtype
                )

        # Validation of `fill_value` will have to be performed
        # before returning self.
        if not self.nullable:
            return self

        return super().fillna(fill_value, method=method)

    def indices_of(
        self, value: ScalarLike
    ) -> cudf.core.column.NumericalColumn:
        return self.codes.indices_of(self._encode(value))

    @property
    def is_monotonic_increasing(self) -> bool:
        return bool(self.ordered) and self.codes.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        return bool(self.ordered) and self.codes.is_monotonic_decreasing

    def as_categorical_column(self, dtype: Dtype) -> CategoricalColumn:
        if isinstance(dtype, str) and dtype == "category":
            return self
        if (
            isinstance(
                dtype, (cudf.core.dtypes.CategoricalDtype, pd.CategoricalDtype)
            )
            and (dtype.categories is None)
            and (dtype.ordered is None)
        ):
            return self

        if isinstance(dtype, pd.CategoricalDtype):
            dtype = CategoricalDtype(
                categories=dtype.categories, ordered=dtype.ordered
            )

        if not isinstance(dtype, CategoricalDtype):
            raise ValueError("dtype must be CategoricalDtype")

        if not isinstance(self.categories, type(dtype.categories._values)):
            # If both categories are of different Column types,
            # return a column full of Nulls.
            return _create_empty_categorical_column(self, dtype)

        return self.set_categories(
            new_categories=dtype.categories, ordered=bool(dtype.ordered)
        )

    def as_numerical_column(self, dtype: Dtype) -> NumericalColumn:
        return self._get_decategorized_column().as_numerical_column(dtype)

    def as_string_column(
        self, dtype, format: str | None = None
    ) -> StringColumn:
        return self._get_decategorized_column().as_string_column(
            dtype, format=format
        )

    def as_datetime_column(
        self, dtype, format: str | None = None
    ) -> DatetimeColumn:
        return self._get_decategorized_column().as_datetime_column(
            dtype, format
        )

    def as_timedelta_column(
        self, dtype, format: str | None = None
    ) -> TimeDeltaColumn:
        return self._get_decategorized_column().as_timedelta_column(
            dtype, format
        )

    def _get_decategorized_column(self) -> ColumnBase:
        if self.null_count == len(self):
            # self.categories is empty; just return codes
            return self.codes
        gather_map = self.codes.astype(libcudf.types.size_type_dtype).fillna(0)
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
    ) -> Optional[Self]:
        out = super()._mimic_inplace(other_col, inplace=inplace)
        if inplace and isinstance(other_col, CategoricalColumn):
            self._codes = other_col._codes
        return out

    def view(self, dtype: Dtype) -> ColumnBase:
        raise NotImplementedError(
            "Categorical column views are not currently supported"
        )

    @staticmethod
    def _concat(
        objs: abc.MutableSequence[CategoricalColumn],
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
        if newsize > libcudf.MAX_COLUMN_SIZE:
            raise MemoryError(
                f"Result of concat cannot have "
                f"size > {libcudf.MAX_COLUMN_SIZE_STR}"
            )
        elif newsize == 0:
            codes_col = column.column_empty(0, head.codes.dtype, masked=True)
        else:
            # Filter out inputs that have 0 length, then concatenate.
            codes = [o for o in codes if len(o)]
            codes_col = libcudf.concat.concat_columns(objs)

        return column.build_categorical_column(
            categories=column.as_column(cats),
            codes=column.build_column(
                codes_col.base_data, dtype=codes_col.dtype
            ),
            mask=codes_col.base_mask,
            size=codes_col.size,
            offset=codes_col.offset,
        )

    def _with_type_metadata(
        self: CategoricalColumn, dtype: Dtype
    ) -> CategoricalColumn:
        if isinstance(dtype, CategoricalDtype):
            return column.build_categorical_column(
                categories=dtype.categories._values,
                codes=column.build_column(
                    self.codes.base_data, dtype=self.codes.dtype
                ),
                mask=self.codes.base_mask,
                ordered=dtype.ordered,
                size=self.codes.size,
                offset=self.codes.offset,
                null_count=self.codes.null_count,
            )
        return self

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
    ) -> CategoricalColumn:
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

            out_col = column.build_categorical_column(
                categories=new_categories,
                codes=self.base_children[0],
                mask=self.base_mask,
                size=self.size,
                offset=self.offset,
                ordered=ordered,
            )
        else:
            out_col = self
            if type(out_col.categories) is not type(new_categories):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                out_col = _create_empty_categorical_column(
                    self,
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
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
        self, new_categories: ColumnBase, ordered=False
    ) -> bool:
        cur_categories = self.categories
        if len(new_categories) != len(cur_categories):
            return False
        if new_categories.dtype != cur_categories.dtype:
            return False
        # if order doesn't matter, sort before the equals call below
        if not ordered:
            cur_categories = cudf.Series(cur_categories).sort_values(
                ignore_index=True
            )
            new_categories = cudf.Series(new_categories).sort_values(
                ignore_index=True
            )
        return cur_categories.equals(new_categories)

    def _set_categories(
        self,
        new_categories: Any,
        is_unique: bool = False,
        ordered: bool = False,
    ) -> CategoricalColumn:
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
            new_cats = cudf.Series(new_cats)._column.unique()

        cur_codes = self.codes
        max_cat_size = (
            len(cur_cats) if len(cur_cats) > len(new_cats) else len(new_cats)
        )
        out_code_dtype = min_unsigned_type(max_cat_size)

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
        new_codes = df._data["new_codes"]

        # codes can't have masks, so take mask out before moving in
        return column.build_categorical_column(
            categories=new_cats,
            codes=column.build_column(
                new_codes.base_data, dtype=new_codes.dtype
            ),
            mask=new_codes.base_mask,
            size=new_codes.size,
            offset=new_codes.offset,
            ordered=ordered,
        )

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
                "items in new_categories are not the same as in "
                "old categories"
            )
        return self._set_categories(new_categories, ordered=ordered)

    def as_ordered(self, ordered: bool):
        if self.dtype.ordered == ordered:
            return self
        return column.build_categorical_column(
            categories=self.categories,
            codes=self.codes,
            mask=self.base_mask,
            size=self.base_size,
            offset=self.offset,
            ordered=ordered,
        )


def _create_empty_categorical_column(
    categorical_column: CategoricalColumn, dtype: "CategoricalDtype"
) -> CategoricalColumn:
    return column.build_categorical_column(
        categories=column.as_column(dtype.categories),
        codes=column.as_column(
            _DEFAULT_CATEGORICAL_VALUE,
            length=categorical_column.size,
            dtype=categorical_column.codes.dtype,
        ),
        offset=categorical_column.offset,
        size=categorical_column.size,
        mask=categorical_column.base_mask,
        ordered=dtype.ordered,
    )


def pandas_categorical_as_column(
    categorical: ColumnLike, codes: Optional[ColumnLike] = None
) -> CategoricalColumn:
    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    codes = categorical.codes if codes is None else codes
    codes = column.as_column(codes)

    valid_codes = codes != codes.dtype.type(_DEFAULT_CATEGORICAL_VALUE)

    mask = None
    if not valid_codes.all():
        mask = bools_to_mask(valid_codes)

    return column.build_categorical_column(
        categories=categorical.categories,
        codes=column.build_column(codes.base_data, codes.dtype),
        size=codes.size,
        mask=mask,
        ordered=categorical.ordered,
    )
