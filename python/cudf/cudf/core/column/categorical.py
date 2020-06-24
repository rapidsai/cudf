# Copyright (c) 2018-2020, NVIDIA CORPORATION.

import pickle

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
import cudf._lib as libcudf
from cudf._lib.transform import bools_to_mask
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.core.dtypes import CategoricalDtype


class CategoricalAccessor(object):
    def __init__(self, column, parent=None):
        """
        Accessor object for categorical properties of the Series values.
        Be aware that assigning to `categories` is a inplace operation,
        while all methods return new categorical data per default.

        Parameters
        ----------
        data : Series or CategoricalIndex

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
        0   null
        1      2
        2      3
        dtype: category
        Categories (2, int64): [2, 3]
        >>> s.cat.set_categories(list('abcde'))
        0   null
        1   null
        2   null
        dtype: category
        Categories (5, object): [a, b, c, d, e]
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
        self._column = column
        self._parent = parent

    @property
    def categories(self):
        """
        The categories of this categorical.
        """
        from cudf.core.index import as_index

        return as_index(self._column.categories)

    @property
    def codes(self):
        """
        Return Series of codes as well as the index.
        """
        return cudf.Series(
            self._column.codes,
            index=self._parent.index if self._parent is not None else None,
        )

    @property
    def ordered(self):
        """
        Whether the categories have an ordered relationship.
        """
        return self._column.ordered

    def as_ordered(self, **kwargs):
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
        out_col = self._column
        if not out_col.ordered:
            kwargs["ordered"] = True
            out_col = self._set_categories(
                self._column.categories, self._column.categories, **kwargs
            )

        return self._return_or_inplace(out_col, **kwargs)

    def as_unordered(self, **kwargs):
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
        out_col = self._column
        if out_col.ordered:
            kwargs["ordered"] = False
            out_col = self._set_categories(
                self._column.categories, self.categories, **kwargs
            )

        return self._return_or_inplace(out_col, **kwargs)

    def add_categories(self, new_categories, **kwargs):
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
        new_categories = column.as_column(new_categories)
        old_categories = self._column.categories

        if (
            old_categories.dtype == "object"
            and new_categories.dtype != "object"
        ) or (
            new_categories.dtype == "object"
            and old_categories.dtype != "object"
        ):
            raise TypeError(
                "cudf does not support mixed types, please type-cast \
                    new_categories to the same type as existing categories."
            )
        common_dtype = np.find_common_type(
            [old_categories.dtype, new_categories.dtype], []
        )

        new_categories = new_categories.astype(common_dtype, copy=False)
        old_categories = old_categories.astype(common_dtype, copy=False)

        if old_categories.isin(new_categories).any():
            raise ValueError("new categories must not include old categories")

        new_categories = old_categories.append(new_categories)
        out_col = self._column
        if not self._categories_equal(new_categories, **kwargs):
            out_col = self._set_categories(
                old_categories, new_categories, **kwargs
            )

        return self._return_or_inplace(out_col, **kwargs)

    def remove_categories(self, removals, **kwargs):
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
        0     10
        1   null
        2   null
        3      2
        4     10
        5      2
        6     10
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
        0   null
        1      1
        2      1
        3      2
        4   null
        5      2
        6   null
        dtype: category
        Categories (2, int64): [1, 2]
        """
        cats = self.categories.to_series()
        removals = cudf.Series(removals, dtype=cats.dtype)
        removals_mask = removals.isin(cats)

        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            vals = removals[~removals_mask].to_array()
            msg = "removals must all be in old categories: {}".format(vals)
            raise ValueError(msg)

        new_categories = cats[~cats.isin(removals)]._column
        out_col = self._column
        if not self._categories_equal(new_categories, **kwargs):
            out_col = self._set_categories(
                self._column.categories, new_categories, **kwargs
            )

        return self._return_or_inplace(out_col, **kwargs)

    def set_categories(self, new_categories, **kwargs):
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

        ordered : bool, default False
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
        0      1
        1      1
        2   null
        3     10
        4   null
        5     10
        dtype: category
        Categories (2, int64): [1, 10]
        >>> s.cat.set_categories([1, 10], inplace=True)
        >>> s
        0      1
        1      1
        2   null
        3     10
        4   null
        5     10
        dtype: category
        Categories (2, int64): [1, 10]
        """
        ordered = kwargs.get("ordered", self.ordered)
        rename = kwargs.pop("rename", False)
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

            out_col = column.build_categorical_column(
                categories=new_categories,
                codes=self._column.base_children[0],
                mask=self._column.base_mask,
                size=self._column.size,
                offset=self._column.offset,
                ordered=ordered,
            )
        else:
            out_col = self._column
            if not (type(out_col.categories) is type(new_categories)):
                # If both categories are of different Column types,
                # return a column full of Nulls.
                out_col = _create_empty_categorical_column(
                    self._column,
                    CategoricalDtype(
                        categories=new_categories, ordered=ordered
                    ),
                )
            elif not self._categories_equal(new_categories, **kwargs):
                out_col = self._set_categories(
                    self._column.categories, new_categories, **kwargs
                )

        return self._return_or_inplace(out_col, **kwargs)

    def reorder_categories(self, new_categories, **kwargs):
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
        out_col = self._set_categories(
            self._column.categories, new_categories, **kwargs
        )

        return self._return_or_inplace(out_col, **kwargs)

    def _categories_equal(self, new_categories, **kwargs):
        cur_categories = self._column.categories
        if len(new_categories) != len(cur_categories):
            return False
        # if order doesn't matter, sort before the equals call below
        if not kwargs.get("ordered", self.ordered):
            cur_categories = cudf.Series(cur_categories).sort_values()
            new_categories = cudf.Series(new_categories).sort_values()
        return cur_categories._column.equals(new_categories._column)

    def _set_categories(self, current_categories, new_categories, **kwargs):
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*.

        Notes
        -----
        Assumes ``new_categories`` is the same dtype as the current categories
        """

        cur_cats = column.as_column(current_categories)
        new_cats = column.as_column(new_categories)

        # Join the old and new categories to build a map from
        # old to new codes, inserting na_sentinel for any old
        # categories that don't exist in the new categories

        # Ensure new_categories is unique first
        if not (kwargs.get("is_unique", False) or new_cats.is_unique):
            # drop_duplicates() instead of unique() to preserve order
            new_cats = (
                cudf.Series(new_cats)
                .drop_duplicates(ignore_index=True)
                ._column
            )

        cur_codes = self.codes
        cur_order = cupy.arange(len(cur_codes))
        old_codes = cupy.arange(len(cur_cats), dtype=cur_codes.dtype)
        new_codes = cupy.arange(len(new_cats), dtype=cur_codes.dtype)

        new_df = cudf.DataFrame({"new_codes": new_codes, "cats": new_cats})
        old_df = cudf.DataFrame({"old_codes": old_codes, "cats": cur_cats})
        cur_df = cudf.DataFrame({"old_codes": cur_codes, "order": cur_order})

        # Join the old and new categories and line up their codes
        df = old_df.merge(new_df, on="cats", how="left")
        # Join the old and new codes to "recode" the codes data buffer
        df = cur_df.merge(df, on="old_codes", how="left")
        df = df.sort_values(by="order")
        df.reset_index(drop=True, inplace=True)

        ordered = kwargs.get("ordered", self.ordered)
        new_codes = df["new_codes"]._column

        # codes can't have masks, so take mask out before moving in
        return column.build_categorical_column(
            categories=new_cats,
            codes=column.as_column(new_codes.base_data, dtype=new_codes.dtype),
            mask=new_codes.base_mask,
            size=new_codes.size,
            offset=new_codes.offset,
            ordered=ordered,
        )

    def _return_or_inplace(self, new_col, **kwargs):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        from cudf.core.index import CategoricalIndex

        owner = self._parent
        inplace = kwargs.get("inplace", False)
        if inplace:
            self._column._mimic_inplace(new_col, inplace=True)
        else:
            if owner is None:
                return new_col
            elif isinstance(owner, CategoricalIndex):
                return CategoricalIndex(new_col, name=owner.name)
            elif isinstance(owner, cudf.Series):
                return cudf.Series(new_col, index=owner.index, name=owner.name)


class CategoricalColumn(column.ColumnBase):
    """Implements operations for Columns of Categorical type
    """

    def __init__(
        self,
        dtype,
        mask=None,
        size=None,
        offset=0,
        null_count=None,
        children=(),
    ):
        """
        Parameters
        ----------
        dtype : CategoricalDtype
        mask : Buffer
            The validity mask
        offset : int
            Data offset
        children : Tuple[Column]
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
    def base_size(self):
        return int(
            (self.base_children[0].size) / self.base_children[0].dtype.itemsize
        )

    def __contains__(self, item):
        try:
            self._encode(item)
        except ValueError:
            return False
        return self._encode(item) in self.as_numerical

    def serialize(self):
        header = {}
        frames = []
        header["type-serialized"] = pickle.dumps(type(self))
        header["dtype"], dtype_frames = self.dtype.serialize()
        header["dtype_frames_count"] = len(dtype_frames)
        frames.extend(dtype_frames)
        header["data"], data_frames = self.codes.serialize()
        header["data_frames_count"] = len(data_frames)
        frames.extend(data_frames)
        if self.nullable:
            mask_header, mask_frames = self.mask.serialize()
            header["mask"] = mask_header
            frames.extend(mask_frames)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        n_dtype_frames = header["dtype_frames_count"]
        dtype = CategoricalDtype.deserialize(
            header["dtype"], frames[:n_dtype_frames]
        )
        n_data_frames = header["data_frames_count"]

        column_type = pickle.loads(header["data"]["type-serialized"])
        data = column_type.deserialize(
            header["data"],
            frames[n_dtype_frames : n_dtype_frames + n_data_frames],
        )
        mask = None
        if "mask" in header:
            mask = Buffer.deserialize(
                header["mask"], [frames[n_dtype_frames + n_data_frames]]
            )
        return column.build_column(
            data=None,
            dtype=dtype,
            mask=mask,
            children=(column.as_column(data.base_data, dtype=data.dtype),),
        )

    def set_base_data(self, value):
        if value is not None:
            raise RuntimeError(
                "CategoricalColumns do not use data attribute of Column, use "
                "`set_base_children` instead"
            )
        else:
            super().set_base_data(value)

    def set_base_mask(self, value):
        super().set_base_mask(value)
        self._codes = None

    def set_base_children(self, value):
        super().set_base_children(value)
        self._codes = None

    @property
    def children(self):
        if self._children is None:
            codes_column = self.base_children[0]

            buf = Buffer(codes_column.base_data)
            buf.ptr = buf.ptr + (self.offset * codes_column.dtype.itemsize)
            buf.size = self.size * codes_column.dtype.itemsize

            codes_column = column.build_column(
                data=buf, dtype=codes_column.dtype, size=self.size,
            )
            self._children = (codes_column,)
        return self._children

    @property
    def as_numerical(self):
        return column.build_column(
            data=self.codes.data, dtype=self.codes.dtype, mask=self.mask
        )

    @property
    def categories(self):
        return self.dtype.categories._values

    @categories.setter
    def categories(self, value):
        self.dtype = CategoricalDtype(
            categories=value, ordered=self.dtype.ordered
        )

    @property
    def codes(self):
        if self._codes is None:
            self._codes = self.children[0].set_mask(self.mask)
        return self._codes

    @property
    def ordered(self):
        return self.dtype.ordered

    @ordered.setter
    def ordered(self, value):
        self.dtype.ordered = value

    def cat(self, parent=None):
        return CategoricalAccessor(self, parent=parent)

    def unary_operator(self, unaryop):
        raise TypeError(
            f"Series of dtype `category` cannot perform the operation: "
            f"{unaryop}"
        )

    def binary_operator(self, op, rhs, reflect=False):

        if not (self.ordered and rhs.ordered) and op not in ("eq", "ne"):
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

    def normalize_binop_value(self, other):
        from cudf.utils import utils

        ary = utils.scalar_broadcast_to(
            self._encode(other), size=len(self), dtype=self.codes.dtype
        )
        col = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=column.as_column(ary),
            mask=self.base_mask,
            ordered=self.dtype.ordered,
        )
        return col

    def sort_by_values(self, ascending=True, na_position="last"):
        codes, inds = self.as_numerical.sort_by_values(ascending, na_position)
        col = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=column.as_column(codes.base_data, dtype=codes.dtype),
            mask=codes.base_mask,
            size=codes.size,
            ordered=self.dtype.ordered,
        )
        return col, inds

    def element_indexing(self, index):
        val = self.as_numerical.element_indexing(index)
        return self._decode(val) if val is not None else val

    @property
    def __cuda_array_interface__(self):
        raise TypeError(
            "Categorical does not support `__cuda_array_interface__`."
            " Please consider using `.codes` or `.categories`"
            " if you need this functionality."
        )

    def to_pandas(self, index=None):
        codes = self.cat().codes.fillna(-1).to_array()
        categories = self.categories.to_pandas()
        data = pd.Categorical.from_codes(
            codes, categories=categories, ordered=self.ordered
        )
        return pd.Series(data, index=index)

    def to_arrow(self):
        return pa.DictionaryArray.from_arrays(
            from_pandas=True,
            ordered=self.ordered,
            indices=self.as_numerical.to_arrow(),
            dictionary=self.categories.to_arrow(),
        )

    def unique(self):
        codes = self.as_numerical.unique()
        return column.build_categorical_column(
            categories=self.categories,
            codes=column.as_column(codes.base_data, dtype=codes.dtype),
            mask=codes.base_mask,
            offset=codes.offset,
            size=codes.size,
            ordered=self.ordered,
        )

    def _encode(self, value):
        return self.categories.find_first_value(value)

    def _decode(self, value):
        if value == self.default_na_value():
            return None
        return self.categories.element_indexing(value)

    def default_na_value(self):
        return -1

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *replacement*.
        """
        replaced = column.as_column(self.cat().codes)

        to_replace_col = column.as_column(
            np.asarray(
                [self._encode(val) for val in to_replace], dtype=replaced.dtype
            )
        )
        replacement_col = column.as_column(
            np.asarray(
                [self._encode(val) for val in replacement],
                dtype=replaced.dtype,
            )
        )

        output = libcudf.replace.replace(
            replaced, to_replace_col, replacement_col
        )

        return column.build_categorical_column(
            categories=self.dtype.categories,
            codes=column.as_column(output.base_data, dtype=output.dtype),
            mask=output.base_mask,
            offset=output.offset,
            size=output.size,
            ordered=self.dtype.ordered,
        )

    def fillna(self, fill_value):
        """
        Fill null values with *fill_value*
        """
        if not self.nullable:
            return self

        fill_is_scalar = np.isscalar(fill_value)

        if fill_is_scalar:
            if fill_value == self.default_na_value():
                fill_value = self.codes.dtype.type(fill_value)
            else:
                try:
                    fill_value = self._encode(fill_value)
                    fill_value = self.codes.dtype.type(fill_value)
                except (ValueError) as err:
                    err_msg = "fill value must be in categories"
                    raise ValueError(err_msg) from err
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
            # TODO: only required if fill_value has a subset of the categories:
            fill_value = fill_value.cat()._set_categories(
                fill_value.cat().categories, self.categories, is_unique=True
            )
            fill_value = column.as_column(fill_value.codes).astype(
                self.codes.dtype
            )

        result = libcudf.replace.replace_nulls(self, fill_value)

        result = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=column.as_column(result.base_data, dtype=result.dtype),
            offset=result.offset,
            size=result.size,
            mask=None,
            ordered=self.dtype.ordered,
        )

        return result

    def find_first_value(self, value, closest=False):
        """
        Returns offset of first value that matches
        """
        return self.as_numerical.find_first_value(self._encode(value))

    def find_last_value(self, value, closest=False):
        """
        Returns offset of last value that matches
        """
        return self.as_numerical.find_last_value(self._encode(value))

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = (
                self.ordered and self.as_numerical.is_monotonic_increasing
            )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = (
                self.ordered and self.as_numerical.is_monotonic_decreasing
            )
        return self._is_monotonic_decreasing

    def as_categorical_column(self, dtype, **kwargs):
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

        if not isinstance(self.categories, type(dtype.categories._values)):
            # If both categories are of different Column types,
            # return a column full of Nulls.
            return _create_empty_categorical_column(self, dtype)

        return self.cat().set_categories(
            new_categories=dtype.categories, ordered=dtype.ordered
        )

    def as_numerical_column(self, dtype, **kwargs):
        return self._get_decategorized_column().as_numerical_column(
            dtype, **kwargs
        )

    def as_string_column(self, dtype, **kwargs):
        return self._get_decategorized_column().as_string_column(
            dtype, **kwargs
        )

    def as_datetime_column(self, dtype, **kwargs):
        return self._get_decategorized_column().as_datetime_column(
            dtype, **kwargs
        )

    def _get_decategorized_column(self):
        if self.null_count == len(self):
            # self.categories is empty; just return codes
            return self.cat().codes._column
        gather_map = self.cat().codes.astype("int32").fillna(0)._column
        out = self.categories.take(gather_map)
        out = out.set_mask(self.mask)
        return out

    def copy(self, deep=True):
        if deep:
            copied_col = libcudf.copying.copy_column(self)

            return column.build_categorical_column(
                categories=self.dtype.categories,
                codes=column.as_column(
                    copied_col.base_data, dtype=copied_col.dtype
                ),
                offset=copied_col.offset,
                size=copied_col.size,
                mask=copied_col.base_mask,
                ordered=self.dtype.ordered,
            )
        else:
            return column.build_categorical_column(
                categories=self.dtype.categories,
                codes=column.as_column(
                    self.codes.base_data, dtype=self.codes.dtype
                ),
                mask=self.base_mask,
                ordered=self.dtype.ordered,
                offset=self.offset,
                size=self.size,
            )

    def __sizeof__(self):
        return (
            self.cat().categories.__sizeof__() + self.cat().codes.__sizeof__()
        )

    def _memory_usage(self, deep=False):
        if deep:
            return self.__sizeof__()
        else:
            return (
                self.categories._memory_usage()
                + self.cat().codes.memory_usage()
            )

    def _mimic_inplace(self, other_col, inplace=False):
        out = super()._mimic_inplace(other_col, inplace=inplace)
        if inplace:
            self._codes = other_col._codes

        return out

    def view(self, dtype):
        raise NotImplementedError(
            "Categorical column views are not currently supported"
        )


def _create_empty_categorical_column(categorical_column, dtype):

    return column.build_categorical_column(
        categories=dtype.categories,
        codes=column.as_column(
            cudf.utils.utils.scalar_broadcast_to(
                categorical_column.default_na_value(),
                categorical_column.size,
                np.dtype(categorical_column.cat().codes),
            )
        ),
        offset=categorical_column.offset,
        size=categorical_column.size,
        mask=categorical_column.base_mask,
        ordered=dtype.ordered,
    )


def pandas_categorical_as_column(categorical, codes=None):

    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    codes = categorical.codes if codes is None else codes
    codes = column.as_column(codes)

    valid_codes = codes.binary_operator("ne", codes.dtype.type(-1))

    mask = None
    if not valid_codes.all():
        mask = bools_to_mask(valid_codes)

    return column.build_categorical_column(
        categories=categorical.categories,
        codes=column.as_column(codes.base_data, dtype=codes.dtype),
        size=codes.size,
        mask=mask,
        ordered=categorical.ordered,
    )
