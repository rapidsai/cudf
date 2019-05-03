# Copyright (c) 2018, NVIDIA CORPORATION.

import pandas as pd
import numpy as np
import pyarrow as pa

from cudf.dataframe import numerical, columnops
from cudf.dataframe.buffer import Buffer
from cudf.utils import utils, cudautils
from cudf.comm.serialize import register_distributed_serializer

import cudf.bindings.replace as cpp_replace
import cudf.bindings.copying as cpp_copying


class CategoricalAccessor(object):
    """
    This mimicks pandas `df.cat` interface.
    """
    def __init__(self, parent, categories, ordered):
        self._parent = parent
        self._categories = tuple(categories)
        self._ordered = ordered

    @property
    def categories(self):
        return self._categories

    @property
    def ordered(self):
        return self._ordered

    @property
    def codes(self):
        from cudf.dataframe.series import Series
        data = self._parent.data
        if self._parent.has_null_mask:
            mask = self._parent.mask
            null_count = self._parent.null_count
            return Series.from_masked_array(data=data.mem, mask=mask.mem,
                                            null_count=null_count)
        else:
            return Series(data)

    def set_categories(self, new_categories):
        """Returns a new Series with the categories set to the
        specified *new_categories*."""
        from cudf.dataframe.series import Series
        col = self._set_categories(new_categories)
        return Series(data=col)

    def _set_categories(self, new_categories):
        """Returns a new CategoricalColumn with the categories set to the
        specified *new_categories*."""
        codemap = {v: i for i, v in enumerate(new_categories)}
        h_recoder = np.zeros(len(self.categories),
                             dtype=self._parent.data.dtype)
        for i, catval in enumerate(self.categories):
            h_recoder[i] = codemap.get(catval, self._parent.default_na_value())
        # recode the data buffer
        recoded = cudautils.recode(self._parent.data.to_gpu_array(), h_recoder,
                                   self._parent.default_na_value())
        buf_rec = Buffer(recoded)
        return self._parent.replace(data=buf_rec, categories=new_categories)


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
        categories = kwargs.pop('categories')
        ordered = kwargs.pop('ordered')
        kwargs.update({'dtype': pd.core.dtypes.dtypes.CategoricalDtype()})
        super(CategoricalColumn, self).__init__(**kwargs)
        self._categories = tuple(categories)
        self._ordered = bool(ordered)

    def serialize(self, serialize):
        header, frames = super(CategoricalColumn, self).serialize(serialize)
        assert 'dtype' not in header
        header['dtype'] = serialize(self._dtype)
        header['categories'] = self._categories
        header['ordered'] = self._ordered
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        data, mask = cls._deserialize_data_mask(deserialize, header, frames)
        dtype = deserialize(*header['dtype'])
        categories = header['categories']
        ordered = header['ordered']
        col = cls(data=data, mask=mask, null_count=header['null_count'],
                  dtype=dtype, categories=categories, ordered=ordered)
        return col

    def _replace_defaults(self):
        params = super(CategoricalColumn, self)._replace_defaults()
        params.update(dict(categories=self._categories,
                           ordered=self._ordered))
        return params

    @property
    def as_numerical(self):
        return self.view(numerical.NumericalColumn, dtype=self.data.dtype)

    def cat(self):
        return CategoricalAccessor(self, categories=self._categories,
                                   ordered=self._ordered)

    def binary_operator(self, binop, rhs):
        msg = 'Series of dtype `category` cannot perform the operation: {}'\
            .format(binop)
        raise TypeError(msg)

    def unary_operator(self, unaryop):
        msg = 'Series of dtype `category` cannot perform the operation: {}'\
            .format(unaryop)
        raise TypeError(msg)

    def unordered_compare(self, cmpop, rhs):
        if not self.is_type_equivalent(rhs):
            raise TypeError('Categoricals can only compare with the same type')
        return self.as_numerical.unordered_compare(cmpop, rhs.as_numerical)

    def ordered_compare(self, cmpop, rhs):
        if not (self._ordered and rhs._ordered):
            msg = "Unordered Categoricals can only compare equality or not"
            raise TypeError(msg)
        if not self.is_type_equivalent(rhs):
            raise TypeError('Categoricals can only compare with the same type')
        return self.as_numerical.ordered_compare(cmpop, rhs.as_numerical)

    def normalize_binop_value(self, other):
        ary = utils.scalar_broadcast_to(self._encode(other), shape=len(self),
                                        dtype=self.data.dtype)
        col = self.replace(data=Buffer(ary), dtype=self.dtype,
                           categories=self._categories, ordered=self._ordered)
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
        data = pd.Categorical.from_codes(codes,
                                         categories=self._categories,
                                         ordered=self._ordered)
        return pd.Series(data, index=index)

    def to_arrow(self):
        indices = self.cat().codes.to_arrow()
        ordered = self.cat()._ordered
        dictionary = pa.array(self.cat().categories)
        return pa.DictionaryArray.from_arrays(
            indices=indices,
            dictionary=dictionary,
            from_pandas=True,
            ordered=ordered
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
        return CategoricalColumn(
            data=Buffer(list(range(0, len(self._categories))),
                        categorical=True),
            categories=self._categories,
            ordered=self._ordered)

    def unique_count(self, method='sort', dropna=True):
        if method != 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, _ = self._unique_segments()
        if dropna is False and self.null_count > 0:
            return len(segs)+1
        return len(segs)

    def value_counts(self, method='sort'):
        if method != 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        segs, sortedvals = self._unique_segments()
        # Return both values and their counts
        out_col = cpp_copying.apply_gather_array(sortedvals, segs)
        out = cudautils.value_count(segs, len(sortedvals))
        out_vals = self.replace(data=out_col.data, mask=None)
        out_counts = numerical.NumericalColumn(data=Buffer(out),
                                               dtype=np.intp)
        return out_vals, out_counts

    def _encode(self, value):
        for i, cat in enumerate(self._categories):
            if cat == value:
                return i
        return -1

    def _decode(self, value):
        for i, cat in enumerate(self._categories):
            if i == value:
                return cat

    def default_na_value(self):
        return -1

    def find_and_replace(self, to_replace, value):
        """
        Return col with *to_replace* replaced with *value*.
        """
        replaced = columnops.as_column(self.cat().codes)

        to_replace_col = columnops.as_column(
            np.asarray([self._encode(val) for val in to_replace],
                       dtype=replaced.dtype)
        )
        value_col = columnops.as_column(
            np.asarray([self._encode(val) for val in value],
                       dtype=replaced.dtype)
        )

        cpp_replace.replace(replaced, to_replace_col, value_col)

        return self.replace(data=replaced.data)

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with *fill_value*
        """
        result = self.copy()

        if np.isscalar(fill_value):
            if fill_value != self.default_na_value():
                if (fill_value not in self.cat().categories):
                    raise ValueError("fill value must be in categories")
            fill_value = pd.Categorical(fill_value,
                                        categories=self.cat().categories)

        fill_value_col = columnops.as_column(
            fill_value, nan_as_null=False)

        # TODO: only required if fill_value has a subset of the categories:
        fill_value_col = fill_value_col.cat()._set_categories(
            self.cat().categories)

        cpp_replace.replace_nulls(result, fill_value_col)

        result = result.replace(mask=None)
        return self._mimic_inplace(result, inplace)


def pandas_categorical_as_column(categorical, codes=None):
    """Creates a CategoricalColumn from a pandas.Categorical

    If ``codes`` is defined, use it instead of ``categorical.codes``
    """
    # TODO fix mutability issue in numba to avoid the .copy()
    codes = (categorical.codes.copy() if codes is None else codes)
    # TODO pending pandas to be improved
    #       https://github.com/pandas-dev/pandas/issues/14711
    #       https://github.com/pandas-dev/pandas/pull/16015
    valid_codes = codes != -1
    buf = Buffer(codes)
    params = dict(data=buf,
                  categories=categorical.categories,
                  ordered=categorical.ordered)
    if not np.all(valid_codes):
        mask = cudautils.compact_mask_bytes(valid_codes)
        nnz = np.count_nonzero(valid_codes)
        null_count = codes.size - nnz
        params.update(dict(mask=Buffer(mask), null_count=null_count))

    return CategoricalColumn(**params)


register_distributed_serializer(CategoricalColumn)
