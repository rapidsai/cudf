# Copyright (c) 2018, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf._lib as libcudf
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.core.dtypes import CategoricalDtype
from cudf.utils import cudautils, utils


class CategoricalAccessor(object):
    """
    This mimicks pandas `df.cat` interface.
    """

    def __init__(self, parent):
        self._parent = parent

    @property
    def categories(self):
        from cudf.core.index import as_index

        return as_index(self._parent._categories)

    @property
    def ordered(self):
        return self._parent._ordered

    @property
    def codes(self):
        from cudf import Series

        data = self._parent.data
        if self._parent.has_null_mask:
            mask = self._parent.mask
            null_count = self._parent.null_count
            return Series.from_masked_array(
                data=data.mem, mask=mask.mem, null_count=null_count
            )
        else:
            return Series(data, name=self._parent.name)

    def as_ordered(self, **kwargs):
        inplace = kwargs.get("inplace", False)
        data = None if inplace else self._parent
        if not self.ordered:
            kwargs["ordered"] = True
            data = self._set_categories(self.categories, **kwargs)
        if data is not None:
            from cudf import Series

            return Series(data=data)

    def as_unordered(self, inplace=False):
        if inplace:
            self._parent._ordered = False
        else:
            from cudf import Series

            return Series(data=self._parent.replace(ordered=False))

    def add_categories(self, new_categories, **kwargs):
        inplace = kwargs.get("inplace", False)
        data = None if inplace else self._parent
        new_categories = column.as_column(new_categories)
        new_categories = self._parent._categories.append(new_categories)
        if not self._categories_equal(new_categories, **kwargs):
            data = self._set_categories(new_categories, **kwargs)
        if data is not None:
            from cudf import Series

            return Series(data=data)

    def remove_categories(self, removals, **kwargs):
        from cudf import Series

        cats = self.categories.to_series()
        removals = Series(removals, dtype=cats.dtype)
        removals_mask = removals.isin(cats)
        # ensure all the removals are in the current categories
        # list. If not, raise an error to match Pandas behavior
        if not removals_mask.all():
            vals = removals[~removals_mask].to_array()
            msg = "removals must all be in old categories: {}".format(vals)
            raise ValueError(msg)
        return self.set_categories(cats[~cats.isin(removals)], **kwargs)

    def set_categories(self, new_categories, **kwargs):
        """Returns a new Series with the categories set to the
        specified *new_categories*."""
        data = self._parent
        new_categories = column.as_column(new_categories)
        # when called with rename=True, the pandas behavior is
        # to replace the current category values with the new
        # categories.
        if kwargs.pop("rename", False):
            # enforce same length
            if len(new_categories) != len(data._categories):
                raise ValueError(
                    "new_categories must have the same "
                    "number of items as old categories"
                )
            elif not kwargs.get("inplace", False):
                # return a copy if inplace=False
                data = data.replace(categories=new_categories, **kwargs)
            else:
                # mutate inplace if inplace=True
                data._categories = new_categories
                ordered = kwargs.get("ordered", self.ordered)
                data._dtype = CategoricalDtype(
                    categories=column.as_column(new_categories),
                    ordered=ordered,
                )
        elif not self._categories_equal(new_categories, **kwargs):
            data = self._set_categories(new_categories, **kwargs)
        if data is not None:
            from cudf import Series

            return Series(data=data)

    def reorder_categories(self, new_categories, **kwargs):
        from cudf.core.series import Series

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
        data = self._set_categories(new_categories, **kwargs)
        if data is not None:
            return Series(data=data)

    def _categories_equal(self, new_categories, **kwargs):
        cur_categories = self._parent._categories
        if len(new_categories) != len(cur_categories):
            return False
        # if order doesn't matter, sort before the equals call below
        if not kwargs.get("ordered", self.ordered):
            from cudf.core.series import Series

            cur_categories = Series(cur_categories).sort_values()
            new_categories = Series(new_categories).sort_values()
        return cur_categories.equals(new_categories)

    def _set_categories(self, new_categories, **kwargs):
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*.

        Notes
        -----
        Assumes ``new_categories`` is the same dtype as the current categories
        """

        from cudf import DataFrame, Series

        cur_cats = self._parent._categories
        new_cats = column.as_column(new_categories)

        # Join the old and new categories to build a map from
        # old to new codes, inserting na_sentinel for any old
        # categories that don't exist in the new categories

        # Ensure new_categories is unique first
        if not (kwargs.get("is_unique", False) or new_cats.is_unique):
            # drop_duplicates() instead of unique() to preserve order
            new_cats = Series(new_cats).drop_duplicates()._column

        cur_codes = self.codes
        cur_order = cudautils.arange(len(cur_codes))
        old_codes = cudautils.arange(len(cur_cats), dtype=cur_codes.dtype)
        new_codes = cudautils.arange(len(new_cats), dtype=cur_codes.dtype)

        new_df = DataFrame({"new_codes": new_codes, "cats": new_cats})
        old_df = DataFrame({"old_codes": old_codes, "cats": cur_cats})
        cur_df = DataFrame({"old_codes": cur_codes, "order": cur_order})

        # Join the old and new categories and line up their codes
        df = old_df.merge(new_df, on="cats", how="left")
        # Join the old and new codes to "recode" the codes data buffer
        df = cur_df.merge(df, on="old_codes", how="left")
        df = df.sort_values(by="order").reset_index(True)

        ordered = kwargs.get("ordered", self.ordered)
        kwargs = df["new_codes"]._column._replace_defaults()
        kwargs.update(categories=new_cats, ordered=ordered, name=None)

        if kwargs.get("inplace", False):
            self._parent._categories = new_cats
            self._parent._mask = kwargs["mask"]
            self._parent._data = kwargs["data"]
            self._parent._ordered = kwargs["ordered"]
            self._parent._null_count = kwargs["null_count"]
            self._parent._dtype = CategoricalDtype(ordered=ordered)
            return None

        return self._parent.replace(**kwargs)


class CategoricalColumn(column.TypedColumnBase):
    """Implements operations for Columns of Categorical type
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        data : Buffer
            The code values
        mask : Buffer; optional
            The validity mask
        null_count : int; optional
            The number of null values in the mask.
        categories : iterable
            The categories
        ordered : bool
            whether the categorical has a logical ordering (e.g. less than)
        """

        ordered = bool(kwargs.pop("ordered"))
        categories = kwargs.pop("categories", [])
        # Default to String dtype if len(categories) == 0, like pandas does
        categories = (
            column.as_column(categories)
            if len(categories) > 0
            else column.column_empty(0, np.dtype("object"), masked=False)
        )

        dtype = CategoricalDtype(
            categories=column.as_column(categories), ordered=ordered
        )
        kwargs.update({"dtype": dtype})
        super(CategoricalColumn, self).__init__(**kwargs)
        self._categories = categories
        self._ordered = ordered

    def __contains__(self, item):
        return self._encode(item) in self.as_numerical

    def serialize(self):
        header, frames = super(CategoricalColumn, self).serialize()
        header["ordered"] = self._ordered
        header["categories"], category_frames = self._categories.serialize()
        header["category_frame_count"] = len(category_frames)
        header["type"] = pickle.dumps(type(self))
        header["dtype"] = self._dtype.str
        frames.extend(category_frames)
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        data, mask = super(CategoricalColumn, cls).deserialize(header, frames)

        # Handle categories that were serialized as a cudf.Column
        category_frames = frames[
            len(frames) - header["category_frame_count"] :
        ]
        cat_typ = pickle.loads(header["categories"]["type"])
        _categories = cat_typ.deserialize(
            header["categories"], category_frames
        )

        categories = column.as_column(_categories)

        return cls(
            data=data,
            mask=mask,
            categories=categories,
            ordered=header["ordered"],
        )

    def _replace_defaults(self):
        params = super(CategoricalColumn, self)._replace_defaults()
        params.update(categories=self._categories, ordered=self._ordered)
        return params

    @property
    def as_numerical(self):
        from cudf.core.column import numerical

        return self.view(numerical.NumericalColumn, dtype=self.data.dtype)

    @property
    def categories(self):
        return self._categories

    def cat(self):
        return CategoricalAccessor(self)

    def binary_operator(self, binop, rhs, reflect=False):
        msg = (
            "Series of dtype `category` cannot perform the operation: "
            "{}".format(binop)
        )
        raise TypeError(msg)

    def unary_operator(self, unaryop):
        msg = (
            "Series of dtype `category` cannot perform the operation: "
            "{}".format(unaryop)
        )
        raise TypeError(msg)

    def unordered_compare(self, cmpop, rhs):
        if not self.is_type_equivalent(rhs):
            raise TypeError("Categoricals can only compare with the same type")
        return self.as_numerical.unordered_compare(cmpop, rhs.as_numerical)

    def ordered_compare(self, cmpop, rhs):
        if not (self._ordered and rhs._ordered):
            msg = "Unordered Categoricals can only compare equality or not"
            raise TypeError(msg)
        if not self.is_type_equivalent(rhs):
            raise TypeError("Categoricals can only compare with the same type")
        return self.as_numerical.ordered_compare(cmpop, rhs.as_numerical)

    def normalize_binop_value(self, other):
        ary = utils.scalar_broadcast_to(
            self._encode(other), shape=len(self), dtype=self.data.dtype
        )
        col = self.replace(
            data=Buffer(ary),
            dtype=self.dtype,
            categories=self._categories,
            ordered=self._ordered,
        )
        return col

    def sort_by_values(self, ascending=True, na_position="last"):
        codes, inds = self.as_numerical.sort_by_values(ascending, na_position)
        return self.replace(data=codes.data), inds

    def element_indexing(self, index):
        val = self.as_numerical.element_indexing(index)
        return self._decode(val) if val is not None else val

    def to_pandas(self, index=None):
        codes = self.cat().codes.fillna(-1).to_array()
        categories = self._categories.to_pandas()
        data = pd.Categorical.from_codes(
            codes, categories=categories, ordered=self._ordered
        )
        return pd.Series(data, index=index)

    def to_arrow(self):
        return pa.DictionaryArray.from_arrays(
            from_pandas=True,
            ordered=self._ordered,
            indices=self.as_numerical.to_arrow(),
            dictionary=self._categories.to_arrow(),
        )

    def unique(self, method=None):
        codes = self.as_numerical.unique(method).data
        return CategoricalColumn(
            data=codes, categories=self._categories, ordered=self._ordered
        )

    def _encode(self, value):
        return self._categories.find_first_value(value)

    def _decode(self, value):
        if value == self.default_na_value():
            return None
        return self._categories.element_indexing(value)

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

        return self.replace(data=output.data)

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with *fill_value*
        """
        if not self.has_null_mask:
            return self

        fill_is_scalar = np.isscalar(fill_value)

        if fill_is_scalar:
            if fill_value == self.default_na_value():
                fill_value = self.data.dtype.type(fill_value)
            else:
                try:
                    fill_value = self._encode(fill_value)
                    fill_value = self.data.dtype.type(fill_value)
                except (ValueError) as err:
                    err_msg = "fill value must be in categories"
                    raise ValueError(err_msg) from err
        else:
            fill_value = column.as_column(fill_value, nan_as_null=False)
            # TODO: only required if fill_value has a subset of the categories:
            fill_value = fill_value.cat()._set_categories(
                self._categories, is_unique=True
            )
            fill_value = column.as_column(fill_value.data).astype(
                self.data.dtype
            )

        result = libcudf.replace.replace_nulls(self, fill_value)

        result = column.build_column(
            result.data, "category", result.mask, categories=self._categories
        )

        return self._mimic_inplace(result.replace(mask=None), inplace)

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        return self.as_numerical.find_first_value(self._encode(value))

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        return self.as_numerical.find_last_value(self._encode(value))

    def searchsorted(self, value, side="left"):
        if not self._ordered:
            raise ValueError("Requires ordered categories")

        value_col = column.as_column(value)
        if not self.is_type_equivalent(value_col):
            raise TypeError("Categoricals can only compare with the same type")

        return libcudf.search.search_sorted(self, value_col, side)

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = (
                self._ordered and self.as_numerical.is_monotonic_increasing
            )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = (
                self._ordered and self.as_numerical.is_monotonic_decreasing
            )
        return self._is_monotonic_decreasing

    def as_categorical_column(self, dtype, **kwargs):
        return self

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
            # self._categories is empty; just return codes
            return self.cat().codes._column
        gather_map = (
            self.cat().codes.astype("int32").fillna(0)._column.data.mem
        )
        out = self._categories.take(gather_map)
        return out.replace(mask=self.mask)

    def copy(self, deep=True):
        """Categorical Columns are immutable, so a deep copy produces a
        copy of the underlying data, mask, categories and a shallow copy
        creates a new column and copies the references of the data, mask
        and categories.
        """
        if deep:
            copied_col = libcudf.copying.copy_column(self)
            category_col = libcudf.copying.copy_column(self._categories)
            return self.replace(
                data=copied_col.data,
                mask=copied_col.mask,
                dtype=self.dtype,
                categories=category_col,
                ordered=self._ordered,
            )
        else:
            params = self._replace_defaults()
            return type(self)(**params)

    def __sizeof__(self):
        return self._categories.__sizeof__() + self.cat().codes.__sizeof__()

    def _memory_usage(self, deep=False):
        if deep:
            return self.__sizeof__()
        else:
            return (
                self._categories._memory_usage()
                + self.cat().codes.memory_usage()
            )


def pandas_categorical_as_column(categorical, codes=None):
    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    # TODO fix mutability issue in numba to avoid the .copy()
    codes = categorical.codes.copy() if codes is None else codes
    # TODO pending pandas to be improved
    #       https://github.com/pandas-dev/pandas/issues/14711
    #       https://github.com/pandas-dev/pandas/pull/16015
    valid_codes = codes != -1
    buf = Buffer(codes)
    params = dict(
        data=buf,
        categories=categorical.categories,
        ordered=categorical.ordered,
    )
    if not np.all(valid_codes):
        mask = cudautils.compact_mask_bytes(valid_codes)
        nnz = np.count_nonzero(valid_codes)
        null_count = codes.size - nnz
        params.update(dict(mask=Buffer(mask), null_count=null_count))

    return CategoricalColumn(**params)
