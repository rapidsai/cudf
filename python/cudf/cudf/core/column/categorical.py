# Copyright (c) 2018, NVIDIA CORPORATION.

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
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

        return as_index(self._parent.categories)

    @property
    def codes(self):
        return cudf.Series(self._parent.codes)

    @property
    def ordered(self):
        return self._parent.ordered

    def as_ordered(self, **kwargs):
        inplace = kwargs.get("inplace", False)
        data = None if inplace else self._parent
        if not self.ordered:
            kwargs["ordered"] = True
            data = self._set_categories(self.categories, **kwargs)
        if data is not None:
            from cudf import Series

            parent = self._parent
            return Series(
                column.build_categorical_column(
                    categories=parent.dtype.categories,
                    codes=parent.cat().codes,
                    mask=parent.mask,
                    ordered=True,
                )
            )

    def as_unordered(self, inplace=False):
        if inplace:
            self._parent.dtype.ordered = False
        else:
            from cudf import Series

            parent = self._parent
            return Series(
                column.build_categorical_column(
                    categories=parent.dtype.categories,
                    codes=parent.codes,
                    mask=parent.mask,
                    ordered=False,
                )
            )

    def add_categories(self, new_categories, **kwargs):
        inplace = kwargs.get("inplace", False)
        data = None if inplace else self._parent
        new_categories = column.as_column(new_categories)
        new_categories = self._parent.categories.append(new_categories)
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
            if len(new_categories) != len(data.categories):
                raise ValueError(
                    "new_categories must have the same "
                    "number of items as old categories"
                )
            elif not kwargs.get("inplace", False):
                # return a copy if inplace=False
                data = data.replace(categories=new_categories, **kwargs)
            else:
                # mutate inplace if inplace=True
                data.categories = new_categories
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
        cur_categories = self._parent.categories
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

        cur_cats = self._parent.categories
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
        new_codes = df["new_codes"]._column
        new_dtype = CategoricalDtype(categories=new_cats, ordered=ordered,)

        if kwargs.get("inplace", False):
            self._parent.data = None
            self._parent.mask = new_codes.mask
            self._parent.dtype = new_dtype
            self._parent.children = (new_codes,)
            return None

        return column.build_column(
            data=None,
            dtype=new_dtype,
            mask=new_codes.mask,
            children=(new_codes,),
        )


class CategoricalColumn(column.ColumnBase):
    """Implements operations for Columns of Categorical type
    """

    def __init__(self, dtype, mask=None, offset=0, children=()):
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
        data = Buffer.empty(0)
        size = children[0].size
        if isinstance(dtype, pd.api.types.CategoricalDtype):
            dtype = CategoricalDtype.from_pandas(dtype)
        if not isinstance(dtype, CategoricalDtype):
            raise ValueError("dtype must be instance of CategoricalDtype")
        super().__init__(
            data,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            children=children,
        )
        self._codes = None

    def __contains__(self, item):
        try:
            self._encode(item)
        except ValueError:
            return False
        return self._encode(item) in self.as_numerical

    def serialize(self):
        header = {}
        frames = []
        header["type"] = pickle.dumps(type(self))
        header["dtype"], dtype_frames = self.dtype.serialize()
        header["dtype_frames_count"] = len(dtype_frames)
        frames.extend(dtype_frames)
        header["data"], data_frames = self.codes.serialize()
        header["data_frames_count"] = len(data_frames)
        frames.extend(data_frames)
        if self.nullable:
            mask_frames = [self.mask_array_view]
        else:
            mask_frames = []
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

        column_type = pickle.loads(header["data"]["type"])
        data = column_type.deserialize(
            header["data"],
            frames[n_dtype_frames : n_dtype_frames + n_data_frames],
        )
        mask = None
        if header["frame_count"] > n_dtype_frames + n_data_frames:
            mask = Buffer(frames[n_dtype_frames + n_data_frames])
        return column.build_column(
            data=None, dtype=dtype, mask=mask, children=(data,)
        )

    @property
    def as_numerical(self):
        return column.build_column(
            data=self.codes.data, dtype=self.codes.dtype, mask=self.mask,
        )

    @property
    def categories(self):
        return self.dtype.categories.as_column()

    @categories.setter
    def categories(self, value):
        self.dtype = CategoricalDtype(
            categories=value, ordered=self.dtype.ordered,
        )

    @property
    def codes(self):
        return self.children[0].set_mask(self.mask)

    @property
    def ordered(self):
        return self.dtype.ordered

    @ordered.setter
    def ordered(self, value):
        self.dtype.ordered = value

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
        if self.dtype != rhs.dtype:
            raise TypeError("Categoricals can only compare with the same type")
        return self.as_numerical.unordered_compare(cmpop, rhs.as_numerical)

    def ordered_compare(self, cmpop, rhs):
        if not (self.ordered and rhs.ordered):
            msg = "Unordered Categoricals can only compare equality or not"
            raise TypeError(msg)
        if self.dtype != rhs.dtype:
            raise TypeError("Categoricals can only compare with the same type")
        return self.as_numerical.ordered_compare(cmpop, rhs.as_numerical)

    def normalize_binop_value(self, other):
        ary = utils.scalar_broadcast_to(
            self._encode(other), size=len(self), dtype=self.codes.dtype
        )
        col = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=column.as_column(ary),
            mask=self.mask,
            ordered=self.dtype.ordered,
        )
        return col

    def sort_by_values(self, ascending=True, na_position="last"):
        codes, inds = self.as_numerical.sort_by_values(ascending, na_position)
        col = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=codes,
            mask=self.mask,
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

    def unique(self, method=None):
        codes = self.as_numerical.unique(method)
        return column.build_categorical_column(
            categories=self.categories,
            codes=codes,
            mask=codes.mask,
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
            codes=output,
            mask=self.mask,
            ordered=self.dtype.ordered,
        )

    def fillna(self, fill_value, inplace=False):
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
                self.categories, is_unique=True
            )
            fill_value = column.as_column(fill_value.codes).astype(
                self.codes.dtype
            )

        result = libcudf.replace.replace_nulls(self, fill_value)

        result = column.build_categorical_column(
            categories=self.dtype.categories,
            codes=result,
            mask=result.mask,
            ordered=self.dtype.ordered,
        )

        result.mask = None
        return self._mimic_inplace(result, inplace)

    def apply_boolean_mask(self, mask):
        codes = super().apply_boolean_mask(mask)
        return column.build_categorical_column(
            categories=self.dtype.categories,
            codes=codes,
            mask=codes.mask,
            ordered=self.dtype.ordered,
        )

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

    def searchsorted(self, value, side="left"):
        if not self.ordered:
            raise ValueError("Requires ordered categories")

        value_col = column.as_column(value)
        if self.dtype != value_col.dtype:
            raise TypeError("Categoricals can only compare with the same type")

        return libcudf.search.search_sorted(self, value_col, side)

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
            # self.categories is empty; just return codes
            return self.cat().codes._column
        gather_map = self.cat().codes.astype("int32").fillna(0)._column
        out = self.categories.take(gather_map)
        out.mask = self.mask
        return out

    def copy(self, deep=True):
        if deep:
            copied_col = libcudf.copying.copy_column(self)
            return column.build_categorical_column(
                categories=self.dtype.categories,
                codes=copied_col,
                mask=copied_col.mask,
                ordered=self.dtype.ordered,
            )
        else:
            return column.build_categorical_column(
                categories=self.dtype.categories,
                codes=self.codes,
                mask=self.mask,
                ordered=self.dtype.ordered,
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
                self._categories._memory_usage()
                + self.cat().codes.memory_usage()
            )


def pandas_categorical_as_column(categorical, codes=None):

    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    codes = categorical.codes if codes is None else codes
    codes = column.as_column(codes)

    valid_codes = codes != -1

    mask = None
    if not np.all(valid_codes):
        mask = cudautils.compact_mask_bytes(valid_codes)
        mask = Buffer(mask)

    return column.build_categorical_column(
        categories=categorical.categories,
        codes=codes,
        mask=mask,
        ordered=categorical.ordered,
    )
