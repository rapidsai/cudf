# Copyright (c) 2018, NVIDIA CORPORATION.


import warnings
from collections import OrderedDict
from numbers import Number

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, is_dict_like

from cudf.utils import cudautils, utils
from cudf import formatting
from .buffer import Buffer
from .index import Index, RangeIndex, as_index
from cudf.settings import NOTSET, settings
from .column import Column
from .datetime import DatetimeColumn
from . import columnops
from cudf.comm.serialize import register_distributed_serializer
from cudf._gdf import nvtx_range_push, nvtx_range_pop


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

    def __init__(self, data=None, index=None, name=None, nan_as_null=True):
        if isinstance(data, pd.Series):
            name = data.name
            index = as_index(data.index)
        if isinstance(data, Series):
            index = data._index
            name = data.name
            data = data._column
        if data is None:
            data = {}

        if not isinstance(data, columnops.TypedColumnBase):
            data = columnops.as_column(data, nan_as_null=nan_as_null)

        if index is not None and not isinstance(index, Index):
            raise TypeError('index not a Index type: got {!r}'.format(index))

        assert isinstance(data, columnops.TypedColumnBase)
        self._column = data
        self._index = RangeIndex(len(data)) if index is None else index
        self.name = name

    @classmethod
    def from_pandas(cls, s, nan_as_null=True):
        return cls(s, nan_as_null=nan_as_null)

    @classmethod
    def from_arrow(cls, s):
        return cls(s)

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
    def shape(self):
        """Returns a tuple representing the dimensionality of the Series.
        """
        return len(self),

    @property
    def dt(self):
        if isinstance(self._column, DatetimeColumn):
            return DatetimeProperties(self)
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
            name=self.name,
        )

    def _copy_construct(self, **kwargs):
        """Shallow copy this object by replacing certain ctor args.
        """
        params = self._copy_construct_defaults()
        cls = type(self)
        params.update(kwargs)
        return cls(**params)

    def copy(self, deep=True):
        result = self._copy_construct()
        if deep:
            result._column = self._column.copy(deep)
        return result

    def __copy__(self, deep=True):
        return self.copy(deep)

    def __deepcopy__(self):
        return self.copy()

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
        index = index if isinstance(index, Index) else as_index(index)
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
        if isinstance(arg, (list, np.ndarray, pd.Series, range,)):
            arg = Series(arg)
        if isinstance(arg, Series):
            if issubclass(arg.dtype.type, np.integer):
                selvals, selinds = columnops.column_select_by_position(
                    self._column, arg)
                index = self.index.take(selinds.to_gpu_array())
            elif arg.dtype in [np.bool, np.bool_]:
                selvals, selinds = columnops.column_select_by_boolmask(
                    self._column, arg)
                index = self.index.take(selinds.to_gpu_array())
            else:
                raise NotImplementedError(arg.dtype)
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
        return self.to_string(nrows=10)

    def __repr__(self):
        return "<cudf.Series nrows={} >".format(len(self))

    def _binaryop(self, other, fn):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other*.  Return the output Series.  The output dtype is
        determined by the input operands.
        """
        nvtx_range_push("PYGDF_BINARY_OP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.binary_operator(fn, other._column)
        result = self._copy_construct(data=outcol)
        nvtx_range_pop()
        return result

    def _rbinaryop(self, other, fn):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other* for reflected operations.  Return the output Series.
        The output dtype is determined by the input operands.
        """
        nvtx_range_push("PYGDF_BINARY_OP", "orange")
        other = self._normalize_binop_value(other)
        outcol = other._column.binary_operator(fn, self._column)
        result = self._copy_construct(data=outcol)
        nvtx_range_pop()
        return result

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

    def __radd__(self, other):
        return self._rbinaryop(other, 'add')

    def __sub__(self, other):
        return self._binaryop(other, 'sub')

    def __rsub__(self, other):
        return self._rbinaryop(other, 'sub')

    def __mul__(self, other):
        return self._binaryop(other, 'mul')

    def __rmul__(self, other):
        return self._rbinaryop(other, 'mul')

    def __pow__(self, other):
        if other == 2:
            return self * self
        else:
            return NotImplemented

    def __floordiv__(self, other):
        return self._binaryop(other, 'floordiv')

    def __rfloordiv__(self, other):
        return self._rbinaryop(other, 'floordiv')

    def __truediv__(self, other):
        if self.dtype in list(truediv_int_dtype_corrections.keys()):
            truediv_type = truediv_int_dtype_corrections[str(self.dtype)]
            return self.astype(truediv_type)._binaryop(other, 'truediv')
        else:
            return self._binaryop(other, 'truediv')

    def __rtruediv__(self, other):
        if self.dtype in list(truediv_int_dtype_corrections.keys()):
            truediv_type = truediv_int_dtype_corrections[str(self.dtype)]
            return self.astype(truediv_type)._rbinaryop(other, 'truediv')
        else:
            return self._rbinaryop(other, 'truediv')

    __div__ = __truediv__

    def _normalize_binop_value(self, other):
        if isinstance(other, Series):
            return other
        else:
            col = self._column.normalize_binop_value(other)
            return self._copy_construct(data=col)

    def _unordered_compare(self, other, cmpops):
        nvtx_range_push("PYGDF_UNORDERED_COMP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.unordered_compare(cmpops, other._column)
        result = self._copy_construct(data=outcol)
        nvtx_range_pop()
        return result

    def _ordered_compare(self, other, cmpops):
        nvtx_range_push("PYGDF_ORDERED_COMP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.ordered_compare(cmpops, other._column)
        result = self._copy_construct(data=outcol)
        nvtx_range_pop()
        return result

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

        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None
        col = Column._concat([o._column for o in objs])
        return cls(data=col, index=index, name=name)

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
        s = self._column.to_pandas(index=index)
        s.name = self.name
        return s

    def to_arrow(self):
        return self._column.to_arrow()

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
    def iloc(self):
        """
        For integer-location based selection.

        Examples
        --------

        >>> sr = Series(list(range(20)))
        # get the value from 1st index
        >>> sr.iloc[1]
        1

        # get the values from 0,2,9 and 18th index
        >>> sr.iloc[0,2,9,18]
        0    0
        2    2
        9    9
        18   18

        # get the values using slice indices
        >>> sr.iloc[3:10:2]
        3    3
        5    5
        7    7
        9    9

        :return:
        Series containing the elements corresponding to the indices
        """
        return Iloc(self)

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

    def argsort(self, ascending=True, na_position="last"):
        """Returns a Series of int64 index that will sort the series.

        Uses Thrust sort.

        Returns
        -------
        result: Series
        """
        return self._sort(ascending=ascending, na_position=na_position)[1]

    def sort_index(self, ascending=True):
        """Sort by the index.
        """
        inds = self.index.argsort(ascending=ascending)
        return self.take(inds.to_gpu_array())

    def sort_values(self, ascending=True, na_position="last"):
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’, ‘last’}, default ‘last’
            'first' puts nulls at the beginning, 'last' puts nulls at the end.
        Returns
        -------
        sorted_obj : cuDF Series

        Difference from pandas:
          * Not supporting: inplace, kind

        Examples
        --------

        .. code-block:: python

              from cudf.dataframe import Series
              s = Series([1,5,2,4,3])
              s.sort_values()

        Output:

        .. code-block:: python

              0    1
              2    2
              4    3
              3    4
              1    5

        """
        if len(self) == 0:
            return self
        vals, inds = self._sort(ascending=ascending, na_position=na_position)
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

    def _sort(self, ascending=True, na_position="last"):
        """
        Sort by values

        Returns
        -------
        2-tuple of key and index
        """
        col_keys, col_inds = self._column.sort_by_values(
            ascending=ascending,
            na_position=na_position
        )
        sr_keys = self._copy_construct(data=col_keys)
        sr_inds = self._copy_construct(data=col_inds)
        return sr_keys, sr_inds

    def replace(self, to_replace, value):
        """
        Replace values given in *to_replace* with *value*.

        Parameters
        ----------
        to_replace : numeric, str or list-like
            Value(s) to replace.

            * numeric or str:

                - values equal to *to_replace* will be replaced with *value*

            * list of numeric or str:

                - If *value* is also list-like, *to_replace* and *value* must
                be of same length.
        value : numeric, str, list-like, or dict
            Value(s) to replace `to_replace` with.

        See also
        --------
        Series.fillna

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.
        """
        if not is_scalar(to_replace):
            if is_scalar(value):
                value = utils.scalar_broadcast_to(
                    value, (len(to_replace),), np.dtype(type(value))
                )
        else:
            if not is_scalar(value):
                raise TypeError(
                    "Incompatible types '{}' and '{}' "
                    "for *to_replace* and *value*.".format(
                        type(to_replace).__name__, type(value).__name__
                    )
                )
            to_replace = [to_replace]
            value = [value]

        if len(to_replace) != len(value):
            raise ValueError(
                "Replacement lists must be"
                "of same length."
                "Expected {}, got {}.".format(len(to_replace), len(value))
            )

        if is_dict_like(to_replace) or is_dict_like(value):
            raise TypeError("Dict-like args not supported in Series.replace()")

        result = self._column.find_and_replace(to_replace, value)

        return self._copy_construct(data=result)

    def reverse(self):
        """Reverse the Series
        """
        data = cudautils.reverse_array(self.to_gpu_array())
        index = as_index(cudautils.reverse_array(self.index.gpu_values))
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
    def count(self, axis=None, skipna=True):
        """The number of non-null values"""
        assert axis in (None, 0) and skipna is True
        return self.valid_count

    def min(self, axis=None, skipna=True):
        """Compute the min of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.min()

    def max(self, axis=None, skipna=True):
        """Compute the max of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.max()

    def sum(self, axis=None, skipna=True):
        """Compute the sum of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.sum()

    def product(self, axis=None, skipna=True):
        """Compute the product of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.product()

    def mean(self, axis=None, skipna=True):
        """Compute the mean of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.mean()

    def std(self, ddof=1, axis=None, skipna=True):
        """Compute the standard deviation of the series
        """
        assert axis in (None, 0) and skipna is True
        return np.sqrt(self.var(ddof=ddof))

    def var(self, ddof=1, axis=None, skipna=True):
        """Compute the variance of the series
        """
        assert axis in (None, 0) and skipna is True
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

    def unique(self, method='sort', sort=True):
        """Returns unique values of this Series.
        default='sort' will be changed to 'hash' when implemented.
        """
        if method is not 'sort':
            msg = 'non sort based unique() not implemented yet'
            raise NotImplementedError(msg)
        if not sort:
            msg = 'not sorted unique not implemented yet.'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return np.empty(0, dtype=self.dtype)
        res = self._column.unique(method=method)
        return Series(res)

    def unique_count(self, method='sort'):
        """Returns the number of unique valies of the Series: approximate version,
        and exact version to be moved to libgdf
        """
        if method is not 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        return self._column.unique_count(method=method)
        # return len(self._column.unique())

    def value_counts(self, method='sort', sort=True):
        """Returns unique values of this Series.
        """
        if method is not 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        vals, cnts = self._column.value_counts(method=method)
        res = Series(cnts, index=as_index(vals))
        if sort:
            return res.sort_values(ascending=False)
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
        return self._copy_construct(data=scaled)

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

    # Misc

    def hash_values(self):
        """Compute the hash of values in this column.
        """
        from . import numerical

        return Series(numerical.column_hash_values(self._column))

    def quantile(self, q, interpolation='midpoint', exact=True,
                 quant_index=True):
        """
        Return values at the given quantile.

        Parameters
        ----------

        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute
        interpolation : {’linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
            This optional parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j:
        columns : list of str
            List of column names to include.
        exact : boolean
            Whether to use approximate or exact quantile algorithm.
        quant_index : boolean
            Whether to use the list of quantiles as index.

        Returns
        -------

        DataFrame

        """
        if not quant_index:
            return Series(self._column.quantile(q, interpolation, exact))
        else:
            return Series(self._column.quantile(q, interpolation, exact),
                          index=as_index(np.asarray(q)))


register_distributed_serializer(Series)


truediv_int_dtype_corrections = {
        'int64': 'float64',
        'int32': 'float32',
        'int': 'float',
}


class DatetimeProperties(object):

    def __init__(self, series):
        self.series = series

    @property
    def year(self):
        return self.get_dt_field('year')

    @property
    def month(self):
        return self.get_dt_field('month')

    @property
    def day(self):
        return self.get_dt_field('day')

    @property
    def hour(self):
        return self.get_dt_field('hour')

    @property
    def minute(self):
        return self.get_dt_field('minute')

    @property
    def second(self):
        return self.get_dt_field('second')

    def get_dt_field(self, field):
        out_column = self.series._column.get_dt_field(field)
        return Series(data=out_column, index=self.series._index)


class Iloc(object):
    """
    For integer-location based selection.
    """

    def __init__(self, sr):
        self._sr = sr

    def __getitem__(self, arg):
        rows = []
        len_idx = len(self._sr)

        if isinstance(arg, tuple):
            for idx in arg:
                rows.append(idx)

        elif isinstance(arg, int):
            rows.append(arg)

        elif isinstance(arg, slice):
            start, stop, step, sln = utils.standard_python_slice(len_idx, arg)
            if sln > 0:
                for idx in range(start, stop, step):
                    rows.append(idx)

        else:
            raise TypeError(type(arg))

        # To check whether all the indices are valid.
        for idx in rows:
            if abs(idx) > len_idx or idx == len_idx:
                raise IndexError("positional indexers are out-of-bounds")

        for i in range(len(rows)):
            if rows[i] < 0:
                rows[i] = len_idx+rows[i]

        # returns the single elem similar to pandas
        if isinstance(arg, int) and len(rows) == 1:
            return self._sr[rows[0]]

        ret_list = []
        for idx in rows:
            ret_list.append(self._sr[idx])

        return Series(ret_list, index=as_index(np.asarray(rows)))

    def __setitem__(self, key, value):
        # throws an exception while updating
        msg = "updating columns using iloc is not allowed"
        raise ValueError(msg)
