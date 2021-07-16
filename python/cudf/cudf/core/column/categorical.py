# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations
from mimetypes import common_types

import pickle
from collections.abc import MutableSequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

import cudf
from cudf import _lib as libcudf
from cudf._lib.scalar import as_device_scalar
from cudf._lib.transform import bools_to_mask
from cudf._lib.categories import (
    set_categories as cpp_set_categories,
    add_categories as cpp_add_categories,
    remove_categories as cpp_remove_categories
)
from cudf._lib.null_mask import create_null_mask, MaskState
from cudf._lib.stream_compaction import drop_duplicates
from cudf._lib.table import Table
from cudf._typing import ColumnLike, Dtype, ScalarLike
from cudf._lib.search import contains
from cudf._lib.copying import gather
from cudf._lib.sort import order_by
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.core.column.methods import ColumnMethods
from cudf.core.dtypes import CategoricalDtype
from cudf.utils.dtypes import (
    find_common_type,
    is_categorical_dtype,
    is_interval_dtype,
    is_mixed_with_object_dtype,
    min_signed_type,
    min_unsigned_type,
)

if TYPE_CHECKING:
    from cudf._typing import SeriesOrIndex
    from cudf.core.column import (
        ColumnBase,
        DatetimeColumn,
        NumericalColumn,
        StringColumn,
        TimeDeltaColumn,
    )


class CategoricalAccessor(ColumnMethods):
    _column: CategoricalColumn

    def __init__(self, parent: SeriesOrIndex):
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
        >>> s
        0    1
        1    2
        2    3
        dtype: category
        Categories (3, int64): [1, 2, 3]
        >>> s.cat.categories
        Int64Index([1, 2, 3], dtype='int64')
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
        if not is_categorical_dtype(parent.dtype):
            raise AttributeError(
                "Can only use .cat accessor with a 'category' dtype"
            )
        super().__init__(parent=parent)

    @property
    def categories(self) -> "cudf.core.index.BaseIndex":
        """
        The categories of this categorical.
        """
        return cudf.core.index.as_index(self._column.categories)

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
        codes = self._column._min_type_codes
        codes = codes.fillna(-1)
        return cudf.Series(codes, index=index)

    @property
    def ordered(self) -> Optional[bool]:
        """
        Whether the categories have an ordered relationship.
        """
        return self._column.ordered

    def as_ordered(self, inplace: bool = False) -> Optional[SeriesOrIndex]:
        """
        Set the Categorical to be ordered.

        Parameters
        ----------

        inplace : bool, default False
            Whether or not to add the categories inplace
            or return a copy of this categorical with
            added categories.

        Returns
        -------
        Categorical
            Ordered Categorical or None if inplace.

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
        >>> s.cat.as_ordered(inplace=True)
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
        """
        return self._return_or_inplace(
            self._column.as_ordered(), inplace=inplace
        )

    def as_unordered(self, inplace: bool = False) -> Optional[SeriesOrIndex]:
        """
        Set the Categorical to be unordered.

        Parameters
        ----------

        inplace : bool, default False
            Whether or not to set the ordered attribute
            in-place or return a copy of this
            categorical with ordered set to False.

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
        >>> s.cat.as_unordered(inplace=True)
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
        return self._return_or_inplace(
            self._column.as_unordered(), inplace=inplace
        )

    def add_categories(
        self, new_categories: Any, inplace: bool = False
    ) -> Optional[SeriesOrIndex]:
        """
        Add new categories.

        `new_categories` will be included at the last/highest
        place in the categories and will be unused directly
        after this call.

        Parameters
        ----------

        new_categories : category or list-like of category
            The new categories to be included.

        inplace : bool, default False
            Whether or not to add the categories inplace
            or return a copy of this categorical with
            added categories.

        Returns
        -------
        cat
            Categorical with new categories added or
            None if inplace.

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
        >>> s.cat.add_categories([0, 3, 4], inplace=True)
        >>> s
        0    1
        1    2
        dtype: category
        Categories (5, int64): [1, 2, 0, 3, 4]
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
        common_dtype = np.find_common_type(
            [old_categories.dtype, new_categories.dtype], []
        )

        new_categories = new_categories.astype(common_dtype)
        old_categories = old_categories.astype(common_dtype)

        if old_categories.isin(new_categories).any():
            raise ValueError("new categories must not include old categories")

        new_categories = old_categories.append(new_categories)
        out_col = self._column
        if not out_col._categories_equal(new_categories):
            out_col = out_col._set_categories(new_categories)

        return self._return_or_inplace(out_col, inplace=inplace)

    def remove_categories(
        self, removals: Any, inplace: bool = False,
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

        inplace : bool, default False
            Whether or not to remove the categories
            inplace or return a copy of this categorical
            with removed categories.

        Returns
        -------
        cat
            Categorical with removed categories or None
            if inplace.

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
        >>> s.cat.remove_categories([10], inplace=True)
        >>> s
        0    <NA>
        1       1
        2       1
        3       2
        4    <NA>
        5       2
        6    <NA>
        dtype: category
        Categories (2, int64): [1, 2]
        """
        removal_column = column.as_column(removals)
        removals_mask = removal_column.isin(self.categories._values)

        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            vals = removal_column[~removals_mask].to_array()
            raise ValueError(f"removals must all be in old categories: {vals}")

        out_col = cpp_remove_categories(self._column, removal_column)

        return self._return_or_inplace(out_col, inplace=inplace)

    def set_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        rename: bool = False,
        inplace: bool = False,
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

        ordered : optional bool, default False
            Whether or not the categorical is treated as
            a ordered categorical. If not given, do
            not change the ordered information.

        rename : bool, default False
            Whether or not the `new_categories` should be
            considered as a rename of the old categories
            or as reordered categories.

        inplace : bool, default False
            Whether or not to reorder the categories in-place
            or return a copy of this categorical with
            reordered categories.

        Returns
        -------
        cat
            Categorical with reordered categories
            or None if inplace.

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
        >>> s.cat.set_categories([1, 10], inplace=True)
        >>> s
        0       1
        1       1
        2    <NA>
        3      10
        4    <NA>
        5      10
        dtype: category
        Categories (2, int64): [1, 10]
        """
        ordered = self.ordered if ordered is None else ordered
        new_categories = column.as_column(new_categories)

        if isinstance(new_categories, CategoricalColumn):
            new_categories = new_categories.categories

        # when called with rename=True, the pandas behavior is
        # to replace the current category values with the new
        # categories.
        if rename:
            # enforce same length
            if len(new_categories) != len(self._column.categories):
                raise ValueError(
                    "new_categories must have the same "
                    "number of items as old categories"
                )

            return column.build_categorical_column(
                categories=new_categories,
                codes=self._column.codes,
                mask=self._column.mask.copy(),
                size=self._column.size,
                offset=0,
                null_count=self._column.null_count,
                ordered=self.ordered
            )


        if not type(self._column.children[1]) == type(new_categories):
            # Return a new categorical column of same size, but null-filled
            out_col = _create_empty_categorical_column(self._column, 
                                                        dtype=CategoricalDtype(
                                                            new_categories,
                                                            ordered=ordered
                                                        )
                                                        )
        elif (
                not self._categories_equal(new_categories, ordered=ordered)
                or not self.ordered == ordered
             ):
                out_col = self._set_categories(
                    new_categories, ordered=ordered,
                )
        else:
            out_col = self._column
        return self._return_or_inplace(out_col, inplace=inplace)

    def reorder_categories(
        self,
        new_categories: Any,
        ordered: bool = False,
        inplace: bool = False,
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


        inplace : bool, default False
            Whether or not to reorder the categories
            inplace or return a copy of this categorical
            with reordered categories.


        Returns
        -------
        cat
            Categorical with reordered categories or
            None if inplace.

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
            inplace=inplace,
        )


class CategoricalColumn(column.ColumnBase):
    """Implements operations for Columns of Categorical type
    """

    dtype: cudf.core.dtypes.CategoricalDtype
    _codes: Optional[NumericalColumn]
    _children: Tuple[NumericalColumn]
    _category_order: Optional[NumericalColumn]

    def __init__(
        self,
        dtype: CategoricalDtype,
        mask: Buffer = None,
        size: int = None,
        offset: int = 0,
        null_count: int = None,
        children: Tuple["column.ColumnBase", ...] = (),
    ):
        """
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
        return self.base_children[0].base_size

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            self._encode(item)
        except ValueError:
            return False
        return self._encode(item) in self.as_numerical

    def serialize(self) -> Tuple[dict, list]:
        """
        The below shows the keys for header dict, with explanation.
        Headers:
            - dtype                 # header for dtype
            - dtype_frames_count    # num frames for `dtype`
            - mask                  # header for mask
            - mask_frames_count     # num frames for `mask`
            - sub_frames_counts     # list of children frames counts
            - subheaders            # children headers
            - sub_frames_types      # type of children
            - frame_count           # number of total frames
        
        The below shows the items stored in `frames` list. The number of
        frames stored for each item can be retrieved via the following key
        from `headers` dict.
        Frames: `length_key_in_header`:
            - dtype_frames: `dtype_frame_count`
            - mask: `mask_frames_count`
            - sub_frame_{i}: `sub_frames_counts[i]`
        """
        header: Dict[Any, Any] = {}
        frames = []
        

        header["type-serialized"] = pickle.dumps(type(self))
        
        header["dtype"], dtype_frames = self.dtype.serialize()
        header["dtype_frames_count"] = len(dtype_frames)
        frames.extend(dtype_frames)

        if self.mask is not None:
            mask_header, mask_frames = self.mask.serialize()
            header["mask"] = mask_header
            header["mask_frames_count"] = len(mask_frames)
            frames.extend(mask_frames)
        
        sub_headers = []
        sub_frames_counts = []
        sub_frame_type = []
        for item in self.children:
            sheader, sframes = item.serialize()
            sub_headers.append(sheader)
            sub_frames_counts.append(len(sframes))
            sub_frame_type.append(type(item))
            frames.extend(sframes)
        
        header["sub_frames_counts"] = sub_frames_counts
        header["sub_frames_types"] = sub_frame_type
        header["subheaders"] = sub_headers
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> CategoricalColumn:
        n_dtype_frames = header["dtype_frames_count"]
        n_sframes = header["sub_frames_counts"]

        b = 0
        dtype = CategoricalDtype.deserialize(
            header["dtype"], frames[b:b+n_dtype_frames]
        )
        b += n_dtype_frames

        mask = None
        if "mask" in header:
            n_mask_frames = header["mask_frames_count"]
            mask = Buffer.deserialize(
                header["mask"], frames[b: b+n_mask_frames]
            )
            b += n_mask_frames

        children = []
        for typ, sheader, n_sframe in zip(header["sub_frames_types"], header["subheaders"], n_sframes):
            child = typ.deserialize(sheader, frames[b:b+n_sframe])
            children.append(child)
            b += n_sframe

        return cast(
            CategoricalColumn,
            column.build_column(
                data=None,
                dtype=dtype,
                mask=mask,
                children=tuple(children),
            ),
        )

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
    def as_numerical(self) -> NumericalColumn:
        return cast(
            cudf.core.column.NumericalColumn,
            column.build_column(
                data=self.codes.data, dtype=self.codes.dtype, mask=self.mask
            ),
        )

    @property
    def categories(self) -> ColumnBase:
        return self.dtype.categories._values

    @categories.setter
    def categories(self, value):
        self.dtype = CategoricalDtype(
            categories=value, ordered=self.dtype.ordered
        )

    @property
    def codes(self) -> NumericalColumn:
        return self.children[0]

    @property
    def _min_type_codes(self) -> NumericalColumn:
        codes = self._column.codes
        codes.set_base_mask(self._column.base_mask)
        dtype = min_signed_type(codes.max(skipna=True))
        codes = codes.astype(dtype=dtype)
        return codes

    @property
    def ordered(self) -> Optional[bool]:
        return self.dtype.ordered

    @ordered.setter
    def ordered(self, value: bool):
        self.dtype.ordered = value

    def unary_operator(self, unaryop: str):
        raise TypeError(
            f"Series of dtype `category` cannot perform the operation: "
            f"{unaryop}"
        )

    def __setitem__(self, key, value):
        if cudf.utils.dtypes.is_scalar(
            value
        ) and cudf._lib.scalar._is_null_host_scalar(value):
            to_add_categories = 0
        else:
            to_add_categories = len(
                cudf.Index(value).difference(self.categories)
            )

        if to_add_categories > 0:
            raise ValueError(
                "Cannot setitem on a Categorical with a new "
                "category, set the categories first"
            )

        if cudf.utils.dtypes.is_scalar(value):
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

    def binary_operator(
        self, op: str, rhs, reflect: bool = False
    ) -> ColumnBase:
        if not (self.ordered and rhs.ordered) and op not in (
            "eq",
            "ne",
            "NULL_EQUALS",
        ):
            if op in ("lt", "gt", "le", "ge"):
                raise TypeError(
                    "Unordered Categoricals can only compare equality or not"
                )
            raise TypeError(
                f"Series of dtype `{self.dtype}` cannot perform the "
                f"operation: {op}"
            )
        if self.dtype != rhs.dtype:
            raise TypeError("Categoricals can only compare with the same type")
        return self.as_numerical.binary_operator(op, rhs.as_numerical)

    def normalize_binop_value(self, other: ScalarLike) -> CategoricalColumn:

        if isinstance(other, np.ndarray) and other.ndim == 0:
            other = other.item()

        codes = cudf.utils.utils.scalar_broadcast_to(
            self._encode(other), size=len(self), dtype=self.codes.dtype
        )
        col = column.build_categorical_column(
            categories=self.dtype.categories._values,
            codes=codes,
            size=self.size,
            ordered=self.dtype.ordered
        )
        return col

    def sort_by_values(
        self, ascending: bool = True, na_position="last"
    ) -> Tuple[CategoricalColumn, NumericalColumn]:
        codes, inds = self.as_numerical.sort_by_values(ascending, na_position)
        col = column.build_categorical_column(
            categories=self.dtype.categories._values,
            codes=column.as_column(codes.base_data, dtype=codes.dtype),
            mask=codes.base_mask,
            size=codes.size,
            ordered=self.dtype.ordered,
        )
        return col, inds

    def element_indexing(self, index: int) -> ScalarLike:
        val = self.as_numerical.element_indexing(index)
        return self._decode(int(val)) if val is not cudf.NA else val

    @property
    def __cuda_array_interface__(self) -> Mapping[str, Any]:
        raise TypeError(
            "Categorical does not support `__cuda_array_interface__`."
            " Please consider using `.codes` or `.categories`"
            " if you need this functionality."
        )

    def to_pandas(self, index: pd.Index = None, **kwargs) -> pd.Series:
        if self.categories.dtype.kind == "f":
            new_mask = bools_to_mask(self.notnull())
            col = self.__class__(dtype=self.dtype,
                                 mask=new_mask,
                                 size=self.size,
                                 offset=self.offset,
                                 null_count=self.null_count,
                                 children=self.base_children)
        else:
            col = self

        signed_dtype = min_signed_type(len(col.categories))
        codes = col.codes.astype(signed_dtype).fillna(-1).to_array()
        if is_interval_dtype(col.categories.dtype):
            # leaving out dropna because it temporarily changes an interval
            # index into a struct and throws off results.
            # TODO: work on interval index dropna
            categories = col.categories.to_pandas()
        else:
            categories = col.categories.dropna(drop_nan=True).to_pandas()
        data = pd.Categorical.from_codes(
            codes, categories=categories, ordered=col.ordered
        )
        return pd.Series(data, index=index)

    def to_arrow(self) -> pa.Array:
        result = super().to_arrow()
        min_type_codes = self._min_type_codes
        return pa.DictionaryArray.from_arrays(indices=min_type_codes.to_arrow(), dictionary=result.dictionary)
        
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

    @property
    def data_array_view(self) -> cuda.devicearray.DeviceNDArray:
        return self.codes.data_array_view

    def _encode(self, value) -> ScalarLike:
        return self.dtype.categories._values.find_first_value(value)

    def _decode(self, value: int) -> ScalarLike:
        if value == self.default_na_value():
            return None
        return self.dtype.categories._values.element_indexing(value)

    def default_na_value(self) -> ScalarLike:
        return -1

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
        replacement_col = column.as_column(replacement)

        if type(to_replace_col) != type(replacement_col):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )

        # create a dataframe containing the pre-replacement categories
        # and a copy of them to work with. The index of this dataframe
        # represents the original ints that map to the categories
        old_cats = cudf.DataFrame()
        old_cats["cats"] = column.as_column(self.dtype.categories)
        new_cats = old_cats.copy(deep=True)

        # Create a column with the appropriate labels replaced
        old_cats["cats_replace"] = old_cats["cats"].replace(
            to_replace_col, replacement_col
        )

        # Construct the new categorical labels
        # If a category is being replaced by an existing one, we
        # want to map it to None. If it's totally new, we want to
        # map it to the new label it is to be replaced by
        dtype_replace = cudf.Series(replacement_col)
        dtype_replace[dtype_replace.isin(old_cats["cats"])] = None
        new_cats["cats"] = new_cats["cats"].replace(
            to_replace_col, dtype_replace
        )

        # anything we mapped to None, we want to now filter out since
        # those categories don't exist anymore
        # Resetting the index creates a column 'index' that associates
        # the original integers to the new labels
        bmask = new_cats["cats"]._column.notna()
        new_cats = cudf.DataFrame(
            {"cats": new_cats["cats"]._column.apply_boolean_mask(bmask)}
        ).reset_index()

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
            self.codes.dtype
        )
        replacement_col = catmap["index"]._column.astype(self.codes.dtype)

        replaced = column.as_column(self.codes)
        output = libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
        )

        return column.build_categorical_column(
            categories=new_cats["cats"]._column,
            codes=column.as_column(output.base_data, dtype=output.dtype),
            mask=output.base_mask,
            offset=output.offset,
            size=output.size,
            ordered=self.dtype.ordered,
        )

    def isnull(self) -> ColumnBase:
        """
        Identify missing values in a CategoricalColumn.
        """
        result = libcudf.unary.is_null(self)

        if self.categories.dtype.kind == "f":
            # Need to consider `np.nan` values incase
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
            # Need to consider `np.nan` values incase
            # of an underlying float column
            categories = libcudf.unary.is_nan(self.categories)
            if categories.any():
                code = self._encode(np.nan)
                result = result & (self.codes != cudf.Scalar(code))

        return result

    def fillna(
        self, fill_value: Any = None, method: Any = None, dtype: Dtype = None
    ) -> CategoricalColumn:
        """
        Fill null values with *fill_value*
        """
        if not self.nullable:
            return self

        if fill_value is not None:
            fill_is_scalar = np.isscalar(fill_value)

            if fill_is_scalar:
                if fill_value == self.default_na_value():
                    fill_value = cudf.Scalar(None, dtype=self.dtype)
                else:
                    fill_value = cudf.Scalar(fill_value, dtype=self.dtype)
            else:
                fill_value = column.as_column(fill_value, nan_as_null=False)
                if isinstance(fill_value, CategoricalColumn):
                    if self.dtype != fill_value.dtype:
                        raise ValueError(
                            "Cannot set a Categorical with another, "
                            "without identical categories"
                        )
                # TODO: only required if fill_value has a subset of the
                # categories:
                fill_value = fill_value._set_categories(
                    self.categories, is_unique=True,
                )

        result = super().fillna(value=fill_value, method=method)

        result = column.build_categorical_column(
            categories=result.children[1],
            codes=result.children[0],
            offset=result.offset,
            size=result.size,
            mask=result.base_mask,
            ordered=self.dtype.ordered,
        )

        return result

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        """
        Returns offset of first value that matches
        """
        return self.as_numerical.find_first_value(self._encode(value))

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        """
        Returns offset of last value that matches
        """
        return self.as_numerical.find_last_value(self._encode(value))

    @property
    def is_monotonic_increasing(self) -> bool:
        return bool(self.ordered) and self.as_numerical.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        return bool(self.ordered) and self.as_numerical.is_monotonic_decreasing

    def as_categorical_column(
        self, dtype: Dtype, **kwargs
    ) -> CategoricalColumn:
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

    def as_numerical_column(self, dtype: Dtype, **kwargs) -> NumericalColumn:
        return self._get_decategorized_column().as_numerical_column(dtype)

    def as_string_column(self, dtype, format=None, **kwargs) -> StringColumn:
        return self._get_decategorized_column().as_string_column(
            dtype, format=format
        )

    def as_datetime_column(self, dtype, **kwargs) -> DatetimeColumn:
        return self._get_decategorized_column().as_datetime_column(
            dtype, **kwargs
        )

    def as_timedelta_column(self, dtype, **kwargs) -> TimeDeltaColumn:
        return self._get_decategorized_column().as_timedelta_column(
            dtype, **kwargs
        )

    def _get_decategorized_column(self) -> ColumnBase:
        if self.null_count == len(self):
            # self.categories is empty; just return codes
            return self.codes
        gather_map = self.codes.astype("int32").fillna(0)
        out = self.categories.take(gather_map)
        out = out.set_mask(self.mask)
        return out

    def __sizeof__(self) -> int:
        return self.categories.__sizeof__() + self.codes.__sizeof__()

    def _memory_usage(self, **kwargs) -> int:
        deep = kwargs.get("deep", False)
        if deep:
            return self.__sizeof__()
        else:
            return self.categories._memory_usage() + self.codes._memory_usage()

    def _mimic_inplace(
        self, other_col: ColumnBase, inplace: bool = False
    ) -> Optional[ColumnBase]:
        out = super()._mimic_inplace(other_col, inplace=inplace)
        if inplace and isinstance(other_col, CategoricalColumn):
            self._codes = other_col._codes
        return out

    def view(self, dtype: Dtype) -> ColumnBase:
        raise NotImplementedError(
            "Categorical column views are not currently supported"
        )

    def _with_type_metadata(
        self: CategoricalColumn, dtype: Dtype
    ) -> CategoricalColumn:
        if isinstance(dtype, CategoricalDtype):
            if len(self.base_children) == 0:
                if dtype._categories is not None:
                    categories = dtype._categories
                else:
                    categories = column.column_empty(0, "int32", masked=False)
                codes = column.column_empty(0, "uint32", masked=False)
            else:
                categories = self.base_children[1]
                codes = self.base_children[0]
            return column.build_categorical_column(
                categories=categories,
                codes=codes,
                mask=self.base_mask,
                ordered=dtype.ordered,
                size=self.size,
                offset=self.offset,
                null_count=self.null_count,
            )
        return self

    def set_categories(
        self, new_categories: Any, ordered: bool = False, rename: bool = False,
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
            if not (type(out_col.categories) is type(new_categories)):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                out_col = _create_empty_categorical_column(
                    self,
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
                    ),
                )
            elif (
                not out_col._categories_equal(new_categories, ordered=ordered)
                or not self.ordered == ordered
            ):
                out_col = out_col._set_categories(
                    new_categories, ordered=ordered,
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
        Assumes ``new_categories`` is the same type as the current categories,
        when both are Numerical type but dtype is different, ``new_categories``
        dtype will direct the output's dtype.
        """
        new_categories = column.as_column(new_categories)

        should_drop_duplicates = is_unique is False or not new_categories.is_unique 
        if should_drop_duplicates:
            new_categories = drop_duplicates(Table({None: new_categories}))._columns[0]

        # libcudf expects source and target categories to have exact same type.
        # Here we upcast the numerical types to make sure that floating point
        # categories can find their integer counterparts.
        # For example: old_categories [1.0, 3.14]
        #              new_categories [3, 2, 1]
        # The codes for "1.0" in old categories should point to "1" in the new
        # categories.
        col = self._column
        new_categories_as_common_type = new_categories
        if isinstance(new_categories, cudf.core.column.NumericalColumn) and not new_categories.dtype == col.children[1].dtype:
            common_type = find_common_type([col.children[1].dtype, new_categories.dtype])
            converted_categories = col.children[1].astype(common_type)
            col = column.build_categorical_column(
                categories=converted_categories,
                codes=col.children[0],
                mask=col.base_mask,
                size=col.base_size,
                offset=col.offset,
                null_count=col.null_count,
                ordered=col.ordered
            )
            new_categories_as_common_type = new_categories.astype(common_type)
        
        res_col = cpp_set_categories(col, new_categories_as_common_type)

        # In case there was an upcast, convert it to result dtype.
        result_cats = res_col.children[1]
        if isinstance(new_categories, cudf.core.column.NumericalColumn) and not result_cats.dtype == new_categories.dtype:
            result_cats = result_cats.astype(new_categories.dtype)
        # Update with `ordered` info
        res_col = column.build_categorical_column(
            categories=result_cats,
            codes=res_col.children[0],
            mask=res_col.base_mask,
            size=res_col.base_size,
            offset=res_col.offset,
            null_count=res_col.null_count,
            ordered=ordered
        )

        return res_col

    def reorder_categories(
        self, new_categories: Any, ordered: bool = False,
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

    def as_ordered(self):
        out_col = self
        if not out_col.ordered:
            out_col = column.build_categorical_column(
                categories=self.categories,
                codes=self.codes,
                mask=self.base_mask,
                size=self.base_size,
                offset=self.offset,
                ordered=True,
            )
        return out_col

    def as_unordered(self):
        out_col = self
        if out_col.ordered:
            out_col = column.build_categorical_column(
                categories=self.categories,
                codes=self.codes,
                mask=self.base_mask,
                size=self.base_size,
                offset=self.offset,
                ordered=False,
            )
        return out_col


def _create_empty_categorical_column(
    categorical_column: CategoricalColumn, dtype: "CategoricalDtype"
) -> CategoricalColumn:
    return column.build_categorical_column(
        categories=column.as_column(dtype.categories),
        codes=column.as_column(
            cudf.utils.utils.scalar_broadcast_to(
                categorical_column.default_na_value(),
                categorical_column.size,
                categorical_column.codes.dtype,
            )
        ),
        offset=categorical_column.offset,
        size=categorical_column.size,
        mask=categorical_column.base_mask,
        ordered=dtype.ordered,
    )


def pandas_categorical_as_column(
    categorical: pd.Categorical, codes: ColumnLike = None
) -> CategoricalColumn:

    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    codes = categorical.codes if codes is None else codes
    codes = column.as_column(codes, dtype="uint32")

    valid_codes = codes != codes.dtype.type(-1)

    mask = None
    if not valid_codes.all():
        mask = bools_to_mask(valid_codes)

    category_column = column.as_column(categorical.dtype.categories)

    return column.build_categorical_column(
        categories=category_column,
        codes=codes,
        size=codes.size,
        mask=mask,
        ordered=categorical.ordered,
    )
