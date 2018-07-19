# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import warnings
from collections import OrderedDict
from numbers import Number

import numpy as np

from . import cudautils, formatting
from .buffer import Buffer
from .index import Index, RangeIndex, GenericIndex
from .settings import NOTSET, settings
from .column import Column
from .datetime import DatetimeColumn, DatetimeProperties
from . import columnops
from .serialize import register_distributed_serializer


class Series(object):
    """
    Data and null-masks.

    ``Series`` objects are used as columns of ``DataFrame``.
    """
    @classmethod
    def from_categorical(cls, categorical, codes=None):
        """Creates from a pandas.Categorical

        If ``codes`` is defined, use it instead of ``categorical.codes``
        """
        from .categorical import pandas_categorical_as_column

        col = pandas_categorical_as_column(categorical, codes=codes)
        return Series(data=col)

    @classmethod
    def from_masked_array(cls, data, mask, null_count=None):
        """Create a Series with null-mask.
        This is equivalent to:

            Series(data).set_mask(mask, null_count=null_count)

        Parameters
        ----------
        data : 1D array-like
            The values.  Null values must not be skipped.  They can appear
            as garbage values.
        mask : 1D array-like of numpy.uint8
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.
        """
        col = columnops.as_column(data).set_mask(mask, null_count=null_count)
        return cls(data=col)

    def __init__(self, data, index=None):
        if isinstance(data, Series):
            index = data._index
            data = data._column
        if not isinstance(data, columnops.TypedColumnBase):
            data = columnops.as_column(data)

        if index is not None and not isinstance(index, Index):
            raise TypeError('index not a Index type: got {!r}'.format(index))

        assert isinstance(data, columnops.TypedColumnBase)
        self._column = data
        self._index = RangeIndex(len(data)) if index is None else index

    def serialize(self, serialize):
        header = {}
        frames = []
        header['index'], index_frames = serialize(self._index)
        frames.extend(index_frames)
        header['index_frame_count'] = len(index_frames)
        header['column'], column_frames = serialize(self._column)
        frames.extend(column_frames)
        header['column_frame_count'] = len(column_frames)
        return header, frames

    @property
    def dt(self):
        if isinstance(self._column, DatetimeColumn):
            return DatetimeProperties(self._column)
        else:
            raise AttributeError("Can only use .dt accessor with datetimelike "
                                 "values")

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        index_nframes = header['index_frame_count']
        index = deserialize(header['index'], frames[:index_nframes])
        frames = frames[index_nframes:]
        column_nframes = header['column_frame_count']
        column = deserialize(header['column'], frames[:column_nframes])
        return Series(column, index=index)

    def _copy_construct_defaults(self):
        return dict(
            data=self._column,
            index=self._index,
        )

    def _copy_construct(self, **kwargs):
        """Shallow copy this object by replacing certain ctor args.
        """
        params = self._copy_construct_defaults()
        cls = type(self)
        params.update(kwargs)
        return cls(**params)

    def reset_index(self):
        """Reset index to RangeIndex
        """
        return self._copy_construct(index=RangeIndex(len(self)))

    def set_index(self, index):
        """Returns a new Series with a different index.

        Parameters
        ----------
        index : Index, Series-convertible
            the new index or values for the new index
        """
        index = index if isinstance(index, Index) else GenericIndex(index)
        return self._copy_construct(index=index)

    def as_index(self):
        return self.set_index(RangeIndex(len(self)))

    def set_mask(self, mask, null_count=None):
        """Create new Series by setting a mask array.

        This will override the existing mask.  The returned Series will
        reference the same data buffer as this Series.

        Parameters
        ----------
        mask : 1D array-like of numpy.uint8
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.

        """
        col = self._column.set_mask(mask, null_count=null_count)
        return self._copy_construct(data=col)

    def __sizeof__(self):
        return self._column.__sizeof__() + self._index.__sizeof__()

    def __len__(self):
        """Returns the size of the ``Series`` including null values.
        """
        return len(self._column)

    def __getitem__(self, arg):
        if isinstance(arg, Series):
            selvals, selinds = columnops.column_select_by_boolmask(
                self._column, arg)
            index = self.index.take(selinds.to_gpu_array())
            return self._copy_construct(data=selvals, index=index)

        elif isinstance(arg, slice):
            index = self.index[arg]         # slice index
            col = self._column[arg]         # slice column
            return self._copy_construct(data=col, index=index)
        elif isinstance(arg, Number):
            # The following triggers a IndexError if out-of-bound
            return self._column.element_indexing(arg)
        else:
            raise NotImplementedError(type(arg))

    def take(self, indices, ignore_index=False):
        """Return Series by taking values from the corresponding *indices*.
        """
        indices = Buffer(indices).to_gpu_array()
        # Handle zero size
        if indices.size == 0:
            return self._copy_construct(data=self.data[:0],
                                        index=self.index[:0])

        data = cudautils.gather(data=self.data.to_gpu_array(), index=indices)

        if self._column.mask:
            mask = self._get_mask_as_series().take(indices).as_mask()
            mask = Buffer(mask)
        else:
            mask = None
        if ignore_index:
            index = RangeIndex(indices.size)
        else:
            index = self.index.take(indices)

        col = self._column.replace(data=Buffer(data), mask=mask)
        return self._copy_construct(data=col, index=index)

    def _get_mask_as_series(self):
        mask = Series(cudautils.ones(len(self), dtype=np.bool))
        if self._column.mask is not None:
            mask = mask.set_mask(self._column.mask).fillna(False)
        return mask

    def __bool__(self):
        """Always raise TypeError when converting a Series
        into a boolean.
        """
        raise TypeError("can't compute boolean for {!r}".format(type(self)))

    def values_to_string(self, nrows=None):
        """Returns a list of string for each element.
        """
        values = self[:nrows]
        out = ['' if v is None else str(v) for v in values]
        return out

    def head(self, n=5):
        return self[:n]

    def to_string(self, nrows=NOTSET):
        """Convert to string

        Parameters
        ----------
        nrows : int
            Maximum number of rows to show.
            If it is None, all rows are shown.
        """
        if nrows is NOTSET:
            nrows = settings.formatting.get(nrows)

        if len(self) == 0:
            return "<empty Series of dtype={}>".format(self.dtype)

        if nrows is None:
            nrows = len(self)
        else:
            nrows = min(nrows, len(self))  # cap row count

        more_rows = len(self) - nrows

        # Prepare cells
        cols = OrderedDict([('', self.values_to_string(nrows=nrows))])
        # Format into a table
        return formatting.format(index=self.index,
                                 cols=cols, more_rows=more_rows)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def _binaryop(self, other, fn):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other*.  Return the output Series.  The output dtype is
        determined by the input operands.
        """
        other = self._normalize_binop_value(other)
        outcol = self._column.binary_operator(fn, other._column)
        return self._copy_construct(data=outcol)

    def _unaryop(self, fn):
        """
        Internal util to call a unary operator *fn* on operands *self*.
        Return the output Series.  The output dtype is determined by the input
        operand.
        """
        outcol = self._column.unary_operator(fn)
        return self._copy_construct(data=outcol)

    def __add__(self, other):
        return self._binaryop(other, 'add')

    def __sub__(self, other):
        return self._binaryop(other, 'sub')

    def __mul__(self, other):
        return self._binaryop(other, 'mul')

    def __floordiv__(self, other):
        return self._binaryop(other, 'floordiv')

    def __truediv__(self, other):
        return self._binaryop(other, 'truediv')

    __div__ = __truediv__

    def _normalize_binop_value(self, other):
        if isinstance(other, Series):
            return other
        else:
            col = self._column.normalize_binop_value(other)
            return self._copy_construct(data=col)

    def _unordered_compare(self, other, cmpops):
        other = self._normalize_binop_value(other)
        outcol = self._column.unordered_compare(cmpops, other._column)
        return self._copy_construct(data=outcol)

    def _ordered_compare(self, other, cmpops):
        other = self._normalize_binop_value(other)
        outcol = self._column.ordered_compare(cmpops, other._column)
        return self._copy_construct(data=outcol)

    def __eq__(self, other):
        return self._unordered_compare(other, 'eq')

    def __ne__(self, other):
        return self._unordered_compare(other, 'ne')

    def __lt__(self, other):
        return self._ordered_compare(other, 'lt')

    def __le__(self, other):
        return self._ordered_compare(other, 'le')

    def __gt__(self, other):
        return self._ordered_compare(other, 'gt')

    def __ge__(self, other):
        return self._ordered_compare(other, 'ge')

    @property
    def cat(self):
        return self._column.cat()

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._column.dtype

    @classmethod
    def _concat(cls, objs, index=True):
        # Concatenate index if not provided
        if index is True:
            index = Index._concat([o.index for o in objs])

        col = Column._concat([o._column for o in objs])
        return cls(data=col, index=index)

    def append(self, arbitrary):
        """Append values from another ``Series`` or array-like object.
        Returns a new copy with the index resetted.
        """
        other = Series(arbitrary)
        other_col = other._column
        # return new series
        return Series(self._column.append(other_col))

    @property
    def valid_count(self):
        """Number of non-null values"""
        return self._column.valid_count

    @property
    def null_count(self):
        """Number of null values"""
        return self._column.null_count

    @property
    def has_null_mask(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._column.has_null_mask

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        data = self._column.fillna(value)
        return self._copy_construct(data=data)

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : str or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._column.to_array(fillna=fillna)

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._column.to_gpu_array(fillna=fillna)

    def to_pandas(self, index=True):
        if index is True:
            index = self.index.to_pandas()
        return self._column.to_pandas(index=index)

    @property
    def data(self):
        """The gpu buffer for the data
        """
        return self._column.data

    @property
    def index(self):
        """The index object
        """
        return self._index

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        return self._column.nullmask

    def as_mask(self):
        """Convert booleans to bitmask

        Returns
        -------
        device array
        """
        return cudautils.compact_mask_bytes(self.to_gpu_array())

    def astype(self, dtype):
        """Convert to the given ``dtype``.

        Returns
        -------
        If the dtype changed, a new ``Series`` is returned by casting each
        values to the given dtype.
        If the dtype is not changed, ``self`` is returned.
        """
        if dtype == self.dtype:
            return self

        return self._copy_construct(data=self._column.astype(dtype))

    def argsort(self, ascending=True):
        """Returns a Series of int64 index that will sort the series.

        Uses stable parallel radixsort.

        Returns
        -------
        result: Series
        """
        return self._sort(ascending=ascending)[1]

    def sort_index(self, ascending=True):
        """Sort by the index.
        """
        inds = self.index.argsort(ascending=ascending)
        return self.take(inds.to_gpu_array())

    def sort_values(self, ascending=True):
        """
        Sort by values.

        Difference from pandas:
        * Support axis='index' only.
        * Not supporting: inplace, kind, na_position

        Details:
        Uses parallel radixsort, which is a stable sort.
        """
        vals, inds = self._sort(ascending=ascending)
        index = self.index.take(inds.to_gpu_array())
        return vals.set_index(index)

    def _n_largest_or_smallest(self, largest, n, keep):
        if not (0 <= n < len(self)):
            raise ValueError("n out-of-bound")
        direction = largest
        if keep == 'first':
            return self.sort_values(ascending=not direction)[:n]
        elif keep == 'last':
            return self.sort_values(ascending=direction)[-n:].reverse()
        else:
            raise ValueError('keep must be either "first", "last"')

    def nlargest(self, n=5, keep='first'):
        """Returns a new Series of the *n* largest element.
        """
        return self._n_largest_or_smallest(n=n, keep=keep, largest=True)

    def nsmallest(self, n=5, keep='first'):
        """Returns a new Series of the *n* smallest element.
        """
        return self._n_largest_or_smallest(n=n, keep=keep, largest=False)

    def _sort(self, ascending=True):
        """
        Sort by values

        Returns
        -------
        2-tuple of key and index
        """
        col_keys, col_inds = self._column.sort_by_values(ascending=ascending)
        sr_keys = self._copy_construct(data=col_keys)
        sr_inds = self._copy_construct(data=col_inds)
        return sr_keys, sr_inds

    def reverse(self):
        """Reverse the Series
        """
        data = cudautils.reverse_array(self.to_gpu_array())
        index = GenericIndex(cudautils.reverse_array(self.index.gpu_values))
        col = self._column.replace(data=Buffer(data))
        return self._copy_construct(data=col, index=index)

    def one_hot_encoding(self, cats, dtype='float64'):
        """Perform one-hot-encoding

        Parameters
        ----------
        cats : sequence of values
                values representing each category.
        dtype : numpy.dtype
                specifies the output dtype.

        Returns
        -------
        A sequence of new series for each category.  Its length is determined
        by the length of ``cats``.
        """
        if self.dtype.kind not in 'iuf':
            raise TypeError('expecting integer or float dtype')

        dtype = np.dtype(dtype)
        out = []
        for cat in cats:
            mask = None  # self.nullmask.to_gpu_array()
            buf = cudautils.apply_equal_constant(
                arr=self.data.to_gpu_array(),
                mask=mask,
                val=cat, dtype=dtype)
            out.append(Series(buf, index=self.index))
        return out

    def label_encoding(self, cats, dtype=None, na_sentinel=-1):
        """Perform label encoding

        Parameters
        ----------
        values : sequence of input values
        dtype: numpy.dtype; optional
               Specifies the output dtype.  If `None` is given, the
               smallest possible integer dtype (starting with np.int32)
               is used.
        na_sentinel : number
            Value to indicate missing category.
        Returns
        -------
        A sequence of encoded labels with value between 0 and n-1 classes(cats)
        """

        if self.null_count != 0:
            mesg = 'series contains NULL values'
            raise ValueError(mesg)

        if self.dtype.kind not in 'iuf':
            raise TypeError('expecting integer or float dtype')

        gpuarr = self.to_gpu_array()
        sr_cats = Series(cats)
        if dtype is None:
            # Get smallest type to represent the category size
            min_dtype = np.min_scalar_type(len(cats))
            # Normalize the size to at least 32-bit
            normalized_sizeof = max(4, min_dtype.itemsize)
            dtype = getattr(np, "int{}".format(normalized_sizeof * 8))
        dtype = np.dtype(dtype)
        labeled = cudautils.apply_label(gpuarr, sr_cats.to_gpu_array(), dtype,
                                        na_sentinel)

        return Series(labeled)

    def factorize(self, na_sentinel=-1):
        """Encode the input values as integer labels

        Parameters
        ----------
        na_sentinel : number
            Value to indicate missing category.

        Returns
        --------
        (labels, cats) : (Series, Series)
            - *labels* contains the encoded values
            - *cats* contains the categories in order that the N-th
              item corresponds to the (N-1) code.
        """
        cats = self.unique()
        labels = self.label_encoding(cats=cats)
        return labels, cats

    # UDF related

    def applymap(self, udf, out_dtype=None):
        """Apply a elemenwise function to transform the values in the Column.

        The user function is expected to take one argument and return the
        result, which will be stored to the output Series.  The function
        cannot reference globals except for other simple scalar objects.

        Parameters
        ----------
        udf : function
            Wrapped by ``numba.cuda.jit`` for call on the GPU as a device
            function.
        out_dtype  : numpy.dtype; optional
            The dtype for use in the output.
            By default, the result will have the same dtype as the source.

        Returns
        -------
        result : Series
            The mask and index are preserved.
        """
        res_col = self._column.applymap(udf, out_dtype=out_dtype)
        return self._copy_construct(data=res_col)

    # Find / Search

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        return self._column.find_first_value(value)

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        return self._column.find_last_value(value)

    #
    # Stats
    #
    def count(self):
        """The number of non-null values"""
        return self.valid_count

    def min(self):
        """Compute the min of the series
        """
        return self._column.min()

    def max(self):
        """Compute the max of the series
        """
        return self._column.max()

    def sum(self):
        """Compute the sum of the series"""
        return self._column.sum()

    def mean(self):
        """Compute the mean of the series
        """
        return self._column.mean()

    def std(self, ddof=1):
        """Compute the standard deviation of the series
        """
        return np.sqrt(self.var(ddof=ddof))

    def var(self, ddof=1):
        """Compute the variance of the series
        """
        mu, var = self.mean_var(ddof=ddof)
        return var

    def mean_var(self, ddof=1):
        """Compute mean and variance at the same time.
        """
        mu, var = self._column.mean_var(ddof=ddof)
        return mu, var

    def sum_of_squares(self):
        return self._column.sum_of_squares()

    def unique_k(self, k):
        warnings.warn("Use .unique() instead", DeprecationWarning)
        return self.unique()

    def unique(self, type='sort'):
        """Returns unique values of this Series.
        default='sort' will be changed to 'hash' when implemented.
        """
        if type is not 'sort':
            msg = 'non sort based unique() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return np.empty(0, dtype=self.dtype)
        res = self._column.unique(type=type)
        return Series(res)

    def unique_count(self, type='sort'):
        """Returns the number of unique valies of the Series: approximate version,
        and exact version to be moved to libgdf
        """
        if type is not 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        return self._column.unique_count(type=type)
        # return len(self._column.unique())

    def value_counts(self, type='sort'):
        """Returns unique values of this Series.
        """
        if type is not 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        vals, cnts = self._column.value_counts(type=type)
        res = Series(cnts, index=GenericIndex(vals))
        return res

    def scale(self):
        """Scale values to [0, 1] in float64
        """
        if self.null_count != 0:
            msg = 'masked series not supported by this operation'
            raise NotImplementedError(msg)
        vmin = self.min()
        vmax = self.max()
        gpuarr = self.to_gpu_array()
        scaled = cudautils.compute_scale(gpuarr, vmin, vmax)
        return Series(scaled)

    # Rounding

    def ceil(self):
        """Rounds each value upward to the smallest integral value not less
        than the original.

        Returns a new Series.
        """
        return self._unaryop('ceil')

    def floor(self):
        """Rounds each value downward to the largest integral value not greater
        than the original.

        Returns a new Series.
        """
        return self._unaryop('floor')


register_distributed_serializer(Series)
