# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.dtypes import CategoricalDtype

import cudf.bindings.copying as cpp_copying
import cudf.bindings.replace as cpp_replace
from cudf.comm.serialize import register_distributed_serializer
from cudf.dataframe import columnops
from cudf.dataframe.buffer import Buffer
from cudf.utils import cudautils, utils


class CategoricalAccessor(object):
    """
    This mimicks pandas `df.cat` interface.
    """

    def __init__(self, parent):
        self._parent = parent

    @property
    def categories(self):
        from cudf.dataframe.index import as_index

        return as_index(self._parent._categories)

    @property
    def ordered(self):
        return self._parent._ordered

    @property
    def codes(self):
        from cudf.dataframe.series import Series

        data = self._parent.data
        if self._parent.has_null_mask:
            mask = self._parent.mask
            null_count = self._parent.null_count
            return Series.from_masked_array(
                data=data.mem, mask=mask.mem, null_count=null_count
            )
        else:
            return Series(data)

    def set_categories(self, new_categories):
        """Returns a new Series with the categories set to the
        specified *new_categories*."""
        from cudf.dataframe.series import Series

        col = self._set_categories(new_categories)
        return Series(data=col)

    def _set_categories(self, new_categories, is_unique=False):
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*.

        Notes
        -----
        Assumes ``new_categories`` is the same dtype as the current categories
        """

        cur_cats = self.categories

        if cur_cats.equals(new_categories):
            return self._parent.copy()

        from cudf import DataFrame

        # Join the old and new categories to build a map from
        # old to new codes, inserting na_sentinel for any old
        # categories that don't exist in the new categories

        new_cats = columnops.as_column(new_categories)

        # Ensure new_categories is unique first
        if not is_unique:
            new_cats = new_cats.unique()

        cur_codes = self.codes
        new_codes = cudautils.arange(len(new_cats), dtype=cur_codes.dtype)
        old_codes = cudautils.arange(len(cur_cats), dtype=cur_codes.dtype)

        cur_df = DataFrame({"old_codes": cur_codes})
        old_df = DataFrame({"old_codes": old_codes, "cats": cur_cats})
        new_df = DataFrame({"new_codes": new_codes, "cats": new_cats})

        # Join the old and new categories and line up their codes
        df = old_df.merge(new_df, on="cats", how="left")
        # Join the old and new codes to "recode" the codes data buffer
        df = cur_df.merge(df, on="old_codes", how="left")

        kwargs = df["new_codes"]._column._replace_defaults()
        kwargs.update(categories=new_cats)
        new_cats._name = None

        return self._parent.replace(**kwargs)


class CategoricalColumn(columnops.TypedColumnBase):
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
            columnops.as_column(categories)
            if len(categories) > 0
            else columnops.column_empty(0, np.dtype("object"), masked=False)
        )

        dtype = CategoricalDtype(ordered=ordered)
        kwargs.update({"dtype": dtype})
        super(CategoricalColumn, self).__init__(**kwargs)
        self._categories = categories
        self._ordered = ordered

    def serialize(self, serialize):
        header, frames = super(CategoricalColumn, self).serialize(serialize)
        header["ordered"] = self._ordered
        header["categories"], category_frames = serialize(self._categories)
        header["category_frame_count"] = len(category_frames)
        frames.extend(category_frames)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        data, mask = super(CategoricalColumn, cls).deserialize(
            deserialize, header, frames
        )
        if "category_frame_count" not in header:
            # Handle data from before categories was a cudf.Column
            categories = header["categories"]
        else:
            # Handle categories that were serialized as a cudf.Column
            categories = frames[len(frames) - header["category_frame_count"] :]
            categories = deserialize(header["categories"], categories)
            categories = columnops.as_column(categories)

        return cls(
            data=data,
            mask=mask,
            categories=categories,
            ordered=header["ordered"],
        )

    def _replace_defaults(self):
        params = super(CategoricalColumn, self)._replace_defaults()
        params.update(dict(categories=self._categories, ordered=self._ordered))
        return params

    @property
    def as_numerical(self):
        from cudf.dataframe import numerical

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

    def astype(self, dtype):
        # custom dtype can't be compared with `==`
        if self.dtype is dtype:
            return self
        return self.as_numerical.astype(dtype)

    def sort_by_values(self, ascending, na_position="last"):
        return self.as_numerical.sort_by_values(ascending, na_position)

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

    def _unique_segments(self):
        """ Common code for unique, unique_count and value_counts"""
        # make dense column
        densecol = self.replace(data=self.to_dense_buffer(), mask=None)
        # sort the column
        sortcol, _ = densecol.sort_by_values(ascending=True)
        # find segments
        sortedvals = sortcol.to_gpu_array()
        segs, begins = cudautils.find_segments(sortedvals)
        return segs, sortedvals

    def unique(self, method=None):
        codes = self.as_numerical.unique(method).data
        return CategoricalColumn(
            data=codes, categories=self._categories, ordered=self._ordered
        )

    def unique_count(self, method="sort", dropna=True):
        if method != "sort":
            msg = "non sort based unique_count() not implemented yet"
            raise NotImplementedError(msg)
        segs, _ = self._unique_segments()
        if dropna is False and self.null_count > 0:
            return len(segs) + 1
        return len(segs)

    def value_counts(self, method="sort"):
        if method != "sort":
            msg = "non sort based value_count() not implemented yet"
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # Return both values and their counts
        out_col = cpp_copying.apply_gather_array(sortedvals, segs)
        out = cudautils.value_count(segs, len(sortedvals))
        out_vals = self.replace(data=out_col.data, mask=None)
        out_counts = columnops.build_column(Buffer(out), np.intp)
        return out_vals, out_counts

    def _encode(self, value):
        return self._categories.find_first_value(value)

    def _decode(self, value):
        return self._categories.element_indexing(value)

    def default_na_value(self):
        return -1

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *replacement*.
        """
        replaced = columnops.as_column(self.cat().codes)

        to_replace_col = columnops.as_column(
            np.asarray(
                [self._encode(val) for val in to_replace], dtype=replaced.dtype
            )
        )
        replacement_col = columnops.as_column(
            np.asarray(
                [self._encode(val) for val in replacement],
                dtype=replaced.dtype,
            )
        )

        output = cpp_replace.replace(replaced, to_replace_col, replacement_col)

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
            fill_value = columnops.as_column(fill_value, nan_as_null=False)
            # TODO: only required if fill_value has a subset of the categories:
            fill_value = fill_value.cat()._set_categories(
                self._categories, is_unique=True
            )
            fill_value = columnops.as_column(fill_value.data).astype(
                self.data.dtype
            )

        result = cpp_replace.apply_replace_nulls(self, fill_value)

        result = columnops.build_column(
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


register_distributed_serializer(CategoricalColumn)
