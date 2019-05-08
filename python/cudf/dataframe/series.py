# Copyright (c) 2018, NVIDIA CORPORATION.


import warnings
from collections import OrderedDict
from numbers import Number

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar, is_dict_like
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from librmm_cffi import librmm as rmm

from cudf.utils import cudautils, utils, ioutils
from cudf import formatting
from cudf.dataframe.buffer import Buffer
from cudf.dataframe.index import Index, RangeIndex, as_index
from cudf.settings import NOTSET, settings
from cudf.dataframe.column import Column
from cudf.dataframe.datetime import DatetimeColumn
from cudf.dataframe import columnops
from cudf.comm.serialize import register_distributed_serializer
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop


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
        from cudf.dataframe.categorical import pandas_categorical_as_column

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

    def __init__(self, data=None, index=None, name=None, nan_as_null=True,
                 dtype=None):
        if isinstance(data, pd.Series):
            name = data.name
            index = as_index(data.index)
        if isinstance(data, Series):
            index = data._index if index is None else index
            name = data.name
            data = data._column
        if data is None:
            data = {}

        if not isinstance(data, columnops.TypedColumnBase):
            data = columnops.as_column(data, nan_as_null=nan_as_null,
                                       dtype=dtype)

        if index is not None and not isinstance(index, Index):
            index = as_index(index)
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

    @property
    def ndim(self):
        """Dimension of the data. Series ndim is always 1.
        """
        return 1

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

    def reset_index(self, drop=False):
        """ Reset index to RangeIndex """
        if not drop:
            return self.to_frame().reset_index(drop=drop)
        else:
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

    def to_frame(self, name=None):
        """Convert Series into a DataFrame

        Parameters
        ----------
        name : str, default None
            Name to be used for the column

        Returns
        -------
        DataFrame
            cudf DataFrame
        """

        from cudf import DataFrame

        if name is not None:
            col = name
        elif self.name is None:
            col = 0
        else:
            col = self.name

        return DataFrame({col: self}, index=self.index)

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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import cudf
        if (method == '__call__' and hasattr(cudf, ufunc.__name__)):
            func = getattr(cudf, ufunc.__name__)
            return func(self)
        else:
            return NotImplemented

    @property
    def empty(self):
        return not len(self)

    def __getitem__(self, arg):
        if isinstance(arg, (list, np.ndarray, pd.Series, range, Index,
                            DeviceNDArray)):
            if len(arg) == 0:
                arg = Series(np.array([], dtype='int32'))
            else:
                arg = Series(arg)
        if isinstance(arg, Series):
            if issubclass(arg.dtype.type, np.integer):
                if self.dtype == np.dtype('object'):
                    idx = arg.to_gpu_array()
                    selvals = self._column[idx]
                    index = self.index.take(idx)
                else:
                    selvals, selinds = columnops.column_select_by_position(
                        self._column, arg)
                    index = self.index.take(selinds.to_gpu_array())
            elif arg.dtype in [np.bool, np.bool_]:
                if self.dtype == np.dtype('object'):
                    idx = cudautils.boolean_array_to_index_array(
                        arg.to_gpu_array())
                    selvals = self._column[idx]
                    index = self.index.take(idx)
                else:
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
        from cudf import Series
        if isinstance(indices, Series):
            indices = indices.to_gpu_array()
        else:
            indices = Buffer(indices).to_gpu_array()
        # Handle zero size
        if indices.size == 0:
            return self._copy_construct(data=self.data[:0],
                                        index=self.index[:0])

        if self.dtype == np.dtype("object"):
            return self[indices]

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
        if self.dtype == np.dtype('object'):
            out = [str(v) for v in values]
        else:
            out = ['' if v is None else str(v) for v in values]
        return out

    def head(self, n=5):
        return self.iloc[:n]

    def tail(self, n=5):
        """
        Returns the last n rows as a new Series

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> print(ser.tail(2))
        3    1
        4    0
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

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

        str_dtype = self.dtype

        if len(self) == 0:
            return "<empty Series of dtype={}>".format(str_dtype)

        if nrows is None:
            nrows = len(self)
        else:
            nrows = min(nrows, len(self))  # cap row count

        more_rows = len(self) - nrows

        # Prepare cells
        cols = OrderedDict([('', self.values_to_string(nrows=nrows))])
        dtypes = OrderedDict([('', self.dtype)])
        # Format into a table
        output = formatting.format(index=self.index,
                                   cols=cols, dtypes=dtypes,
                                   more_rows=more_rows,
                                   series_spacing=True)
        return output + "\nName: {}, dtype: {}".format(self.name, str_dtype)\
            if self.name is not None else output + \
            "\ndtype: {}".format(str_dtype)

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
        from cudf import DataFrame
        if isinstance(other, DataFrame):
            return other._binaryop(self, fn)
        nvtx_range_push("CUDF_BINARY_OP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.binary_operator(fn, other._column)
        result = self._copy_construct(data=outcol)
        result.name = None
        nvtx_range_pop()
        return result

    def _rbinaryop(self, other, fn):
        """
        Internal util to call a binary operator *fn* on operands *self*
        and *other* for reflected operations.  Return the output Series.
        The output dtype is determined by the input operands.
        """
        from cudf import DataFrame
        if isinstance(other, DataFrame):
            return other._binaryop(self, fn)
        nvtx_range_push("CUDF_BINARY_OP", "orange")
        other = self._normalize_binop_value(other)
        outcol = other._column.binary_operator(fn, self._column)
        result = self._copy_construct(data=outcol)
        result.name = None
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

    def __mod__(self, other):
        return self._binaryop(other, 'mod')

    def __rmod__(self, other):
        return self._rbinaryop(other, 'mod')

    def __pow__(self, other):
        return self._binaryop(other, 'pow')

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

    def _bitwise_binop(self, other, op):
        if (np.issubdtype(self.dtype.type, np.integer) and
                np.issubdtype(other.dtype.type, np.integer)):
            return self._binaryop(other, op)
        else:
            raise TypeError(
                f"Operation 'bitwise {op}' not supported between "
                f"{self.dtype.type.__name__} and {other.dtype.type.__name__}"
            )

    def __and__(self, other):
        """Performs vectorized bitwise and (&) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, 'and')

    def __or__(self, other):
        """Performs vectorized bitwise or (|) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, 'or')

    def __xor__(self, other):
        """Performs vectorized bitwise xor (^) on corresponding elements of two
        series.
        """
        return self._bitwise_binop(other, 'xor')

    def _normalize_binop_value(self, other):
        if isinstance(other, Series):
            return other
        elif isinstance(other, Index):
            return Series(other)
        else:
            col = self._column.normalize_binop_value(other)
            return self._copy_construct(data=col)

    def _unordered_compare(self, other, cmpops):
        nvtx_range_push("CUDF_UNORDERED_COMP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.unordered_compare(cmpops, other._column)
        result = self._copy_construct(data=outcol)
        result.name = None
        nvtx_range_pop()
        return result

    def _ordered_compare(self, other, cmpops):
        nvtx_range_push("CUDF_ORDERED_COMP", "orange")
        other = self._normalize_binop_value(other)
        outcol = self._column.ordered_compare(cmpops, other._column)
        result = self._copy_construct(data=outcol)
        result.name = None
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

    def __invert__(self):
        """Bitwise invert (~)/(not) for each element

        Returns a new Series.
        """
        if np.issubdtype(self.dtype.type, np.integer):
            return self._unaryop('not')
        else:
            raise TypeError(
                f"Operation `~` not supported on {self.dtype.type.__name__}"
            )

    def __neg__(self):
        """Negatated value (-) for each element

        Returns a new Series.
        """
        return self.__mul__(-1)

    @property
    def cat(self):
        return self._column.cat()

    @property
    def str(self):
        return self._column.str(self.index)

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._column.dtype

    @classmethod
    def _concat(cls, objs, axis=0, index=True):
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

    def masked_assign(self, value, mask):
        """Assign a scalar value to a series using a boolean mask
        df[df < 0] = 0

        Parameters
        ----------
        value : scalar
            scalar value for assignment
        mask : cudf Series
            Boolean Series

        Returns
        -------
        cudf Series
            cudf series with new value set to where mask is True
        """

        data = self._column.masked_assign(value, mask)
        return self._copy_construct(data=data)

    def fillna(self, value, method=None, axis=None, inplace=False, limit=None):
        """Fill null values with ``value``.

        Parameters
        ----------
        value : scalar or Series-like
            Value to use to fill nulls. If Series-like, null values
            are filled with the values in corresponding indices of the
            given Series.

        Returns
        -------
        result : Series
            Copy with nulls filled.
        """
        if method is not None:
            raise NotImplementedError("The method keyword is not supported")
        if limit is not None:
            raise NotImplementedError("The limit keyword is not supported")
        if axis:
            raise NotImplementedError("The axis keyword is not supported")

        data = self._column.fillna(value, inplace=inplace)

        if not inplace:
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

    def isnull(self):
        """Identify missing values in a Series.
        """
        if not self.has_null_mask:
            return Series(cudautils.zeros(len(self), np.bool_), name=self.name,
                          index=self.index)

        mask = cudautils.isnull_mask(self.data, self.nullmask.to_gpu_array())
        return Series(mask, name=self.name, index=self.index)

    def isna(self):
        """Identify missing values in a Series. Alias for isnull.
        """
        return self.isnull()

    def notna(self):
        """Identify non-missing values in a Series.
        """
        if not self.has_null_mask:
            return Series(cudautils.ones(len(self), np.bool_), name=self.name,
                          index=self.index)

        mask = cudautils.notna_mask(self.data, self.nullmask.to_gpu_array())
        return Series(mask, name=self.name, index=self.index)

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

    @index.setter
    def index(self, _index):
        self._index = _index

    @property
    def iloc(self):
        """
        For integer-location based selection.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(list(range(20)))

        Get the value from 1st index

        >>> sr.iloc[1]
        1

        Get the values from 0,2,9 and 18th index

        >>> sr.iloc[0,2,9,18]
         0    0
         2    2
         9    9
        18   18

        Get the values using slice indices

        >>> sr.iloc[3:10:2]
        3    3
        5    5
        7    7
        9    9

        Returns
        -------
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
        >>> import cudf
        >>> s = cudf.Series([1, 5, 2, 4, 3])
        >>> s.sort_values()
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

    def min(self, axis=None, skipna=True, dtype=None):
        """Compute the min of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.min(dtype=dtype)

    def max(self, axis=None, skipna=True, dtype=None):
        """Compute the max of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.max(dtype=dtype)

    def sum(self, axis=None, skipna=True, dtype=None):
        """Compute the sum of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.sum(dtype=dtype)

    def product(self, axis=None, skipna=True, dtype=None):
        """Compute the product of the series"""
        assert axis in (None, 0) and skipna is True
        return self._column.product(dtype=dtype)

    def cummin(self, axis=0, skipna=True):
        """Compute the cumulative minimum of the series"""
        assert axis in (None, 0) and skipna is True
        return Series(self._column._apply_scan_op('min'), name=self.name,
                      index=self.index)

    def cummax(self, axis=0, skipna=True):
        """Compute the cumulative maximum of the series"""
        assert axis in (None, 0) and skipna is True
        return Series(self._column._apply_scan_op('max'), name=self.name,
                      index=self.index)

    def cumsum(self, axis=0, skipna=True):
        """Compute the cumulative sum of the series"""
        assert axis in (None, 0) and skipna is True

        # pandas always returns int64 dtype if original dtype is int
        if np.issubdtype(self.dtype, np.integer):
            return Series(self.astype(np.int64)._column._apply_scan_op('sum'),
                          name=self.name, index=self.index)
        else:
            return Series(self._column._apply_scan_op('sum'), name=self.name,
                          index=self.index)

    def cumprod(self, axis=0, skipna=True):
        """Compute the cumulative product of the series"""
        assert axis in (None, 0) and skipna is True

        # pandas always returns int64 dtype if original dtype is int
        if np.issubdtype(self.dtype, np.integer):
            return Series(
                self.astype(np.int64)._column._apply_scan_op('product'),
                name=self.name, index=self.index)
        else:
            return Series(self._column._apply_scan_op('product'),
                          name=self.name, index=self.index)

    def mean(self, axis=None, skipna=True, dtype=None):
        """Compute the mean of the series
        """
        assert axis in (None, 0) and skipna is True
        return self._column.mean(dtype=dtype)

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

    def sum_of_squares(self, dtype=None):
        return self._column.sum_of_squares(dtype=dtype)

    def unique_k(self, k):
        warnings.warn("Use .unique() instead", DeprecationWarning)
        return self.unique()

    def unique(self, method='sort', sort=True):
        """Returns unique values of this Series.
        default='sort' will be changed to 'hash' when implemented.
        """
        if method != 'sort':
            msg = 'non sort based unique() not implemented yet'
            raise NotImplementedError(msg)
        if not sort:
            msg = 'not sorted unique not implemented yet.'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return np.empty(0, dtype=self.dtype)
        res = self._column.unique(method=method)
        return Series(res)

    def nunique(self, method='sort', dropna=True):
        """Returns the number of unique values of the Series: approximate version,
        and exact version to be moved to libgdf
        """
        if method != 'sort':
            msg = 'non sort based unique_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return 0
        return self._column.unique_count(method=method, dropna=dropna)
        # return len(self._column.unique())

    def value_counts(self, method='sort', sort=True):
        """Returns unique values of this Series.
        """
        if method != 'sort':
            msg = 'non sort based value_count() not implemented yet'
            raise NotImplementedError(msg)
        if self.null_count == len(self):
            return Series(np.array([], dtype=np.int64))
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

    # Absolute
    def abs(self):
        """Absolute value of each element of the series.

        Returns a new Series.
        """
        return self._unaryop('abs')

    def __abs__(self):
        return self.abs()

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

    # Math
    def _float_math(self, op):
        if np.issubdtype(self.dtype.type, np.floating):
            return self._unaryop(op)
        else:
            raise TypeError(
                f"Operation '{op}' not supported on {self.dtype.type.__name__}"
            )

    def sin(self):
        return self._float_math('sin')

    def cos(self):
        return self._float_math('cos')

    def tan(self):
        return self._float_math('tan')

    def asin(self):
        return self._float_math('asin')

    def acos(self):
        return self._float_math('acos')

    def atan(self):
        return self._float_math('atan')

    def exp(self):
        return self._float_math('exp')

    def log(self):
        return self._float_math('log')

    def sqrt(self):
        return self._unaryop('sqrt')

    # Misc

    def hash_values(self):
        """Compute the hash of values in this column.
        """
        from cudf.dataframe import numerical

        return Series(numerical.column_hash_values(self._column))

    def hash_encode(self, stop, use_name=False):
        """Encode column values as ints in [0, stop) using hash function.

        Parameters
        ----------
        stop : int
            The upper bound on the encoding range.
        use_name : bool
            If ``True`` then combine hashed column values
            with hashed column name. This is useful for when the same
            values in different columns should be encoded
            with different hashed values.
        Returns
        -------
        result: Series
            The encoded Series.
        """
        assert stop > 0

        from cudf.dataframe import numerical
        initial_hash = np.asarray(hash(self.name)) if use_name else None
        hashed_values = numerical.column_hash_values(
            self._column, initial_hash_values=initial_hash)

        # TODO: Binary op when https://github.com/rapidsai/cudf/pull/892 merged
        mod_vals = cudautils.modulo(hashed_values.data.to_gpu_array(), stop)
        return Series(mod_vals)

    def quantile(self, q=0.5, interpolation='linear', exact=True,
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

    def describe(self, percentiles=None, include=None, exclude=None):
        """Compute summary statistics of a Series. For numeric
        data, the output includes the minimum, maximum, mean, median,
        standard deviation, and various quantiles. For object data, the output
        includes the count, number of unique values, the most common value, and
        the number of occurrences of the most common value.

        Parameters
        ----------
        percentiles : list-like, optional
            The percentiles used to generate the output summary statistics.
            If None, the default percentiles used are the 25th, 50th and 75th.
            Values should be within the interval [0, 1].

        Returns
        -------
        A DataFrame containing summary statistics of relevant columns from
        the input DataFrame.

        Examples
        --------
        Describing a ``Series`` containing numeric values.
        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> print(s.describe())
           stats   values
        0  count     10.0
        1   mean      5.5
        2    std  3.02765
        3    min      1.0
        4    25%      2.5
        5    50%      5.5
        6    75%      7.5
        7    max     10.0
        """

        from cudf import DataFrame

        def _prepare_percentiles(percentiles):
            percentiles = list(percentiles)

            if not all(0 <= x <= 1 for x in percentiles):
                raise ValueError("All percentiles must be between 0 and 1, "
                                 "inclusive.")

            # describe always includes 50th percentile
            if 0.5 not in percentiles:
                percentiles.append(0.5)

            percentiles = np.sort(percentiles)
            return percentiles

        def _format_percentile_names(percentiles):
            return ['{0}%'.format(int(x*100)) for x in percentiles]

        def _format_stats_values(stats_data):
            return list(map(lambda x: round(x, 6), stats_data))

        def describe_numeric(self):
            # mimicking pandas
            names = ['count', 'mean', 'std', 'min'] + \
                    _format_percentile_names(percentiles) + ['max']
            data = [self.count(), self.mean(), self.std(), self.min()] + \
                self.quantile(percentiles).to_array().tolist() + [self.max()]
            data = _format_stats_values(data)

            values_name = 'values'
            if self.name:
                values_name = self.name

            return DataFrame({'stats': names, values_name: data})

        def describe_categorical(self):
            # blocked by StringColumn/DatetimeColumn support for
            # value_counts/unique
            pass

        if percentiles is not None:
            percentiles = _prepare_percentiles(percentiles)
        else:
            # pandas defaults
            percentiles = np.array([0.25, 0.5, 0.75])

        if np.issubdtype(self.dtype, np.number):
            return describe_numeric(self)
        else:
            raise NotImplementedError("Describing non-numeric columns is not "
                                      "yet supported")

    def digitize(self, bins, right=False):
        """Return the indices of the bins to which each value in series belongs.

        Notes
        -----
        Monotonicity of bins is assumed and not checked.

        Parameters
        ----------
        bins : np.array
            1-D monotonically, increasing array with same type as this series.
        right : bool
            Indicates whether interval contains the right or left bin edge.

        Returns
        -------
        A new Series containing the indices.
        """
        from cudf.dataframe import numerical

        return Series(numerical.digitize(self._column, bins, right))

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values of an input array by periods positions and store the
        output in a new array.

        Notes
        -----
        Shift currently only supports float and integer dtype columns with
        no null values.
        """
        assert axis in (None, 0) and freq is None and fill_value is None

        if self.null_count != 0:
            raise AssertionError("Shift currently requires columns with no "
                                 "null values")

        if not np.issubdtype(self.dtype, np.number):
            raise NotImplementedError("Shift currently only supports "
                                      "numeric dtypes")
        if periods == 0:
            return self

        input_dary = self.data.to_gpu_array()
        output_dary = rmm.device_array_like(input_dary)
        cudautils.gpu_shift.forall(output_dary.size)(input_dary, output_dary,
                                                     periods)
        return Series(output_dary, name=self.name, index=self.index)

    def diff(self, periods=1):
        """Calculate the difference between values at positions i and i - N in
        an array and store the output in a new array.
        Notes
        -----
        Diff currently only supports float and integer dtype columns with
        no null values.
        """
        if self.null_count != 0:
            raise AssertionError("Diff currently requires columns with no "
                                 "null values")

        if not np.issubdtype(self.dtype, np.number):
            raise NotImplementedError("Diff currently only supports "
                                      "numeric dtypes")

        input_dary = self.data.to_gpu_array()
        output_dary = rmm.device_array_like(input_dary)
        cudautils.gpu_diff.forall(output_dary.size)(input_dary, output_dary,
                                                    periods)
        return Series(output_dary, name=self.name, index=self.index)

    def groupby(self, group_series=None, level=None, sort=False,
                group_keys=True):
        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )

        from cudf.groupby.groupby import SeriesGroupBy
        return SeriesGroupBy(self, group_series, level, sort)

    def to_json(self, path_or_buf=None, *args, **kwargs):
        """
        Convert the cuDF object to a JSON string.
        Note nulls and NaNs will be converted to null and datetime objects
        will be converted to UNIX timestamps.
        Parameters
        ----------
        path_or_buf : string or file handle, optional
            File path or object. If not specified, the result is returned as
            a string.
        orient : string
            Indication of expected JSON string format.
            * Series
                - default is 'index'
                - allowed values are: {'split','records','index','table'}
            * DataFrame
                - default is 'columns'
                - allowed values are:
                {'split','records','index','columns','values','table'}
            * The format of the JSON string
                - 'split' : dict like {'index' -> [index],
                'columns' -> [columns], 'data' -> [values]}
                - 'records' : list like
                [{column -> value}, ... , {column -> value}]
                - 'index' : dict like {index -> {column -> value}}
                - 'columns' : dict like {column -> {index -> value}}
                - 'values' : just the values array
                - 'table' : dict like {'schema': {schema}, 'data': {data}}
                describing the data, and the data component is
                like ``orient='records'``.
        date_format : {None, 'epoch', 'iso'}
            Type of date conversion. 'epoch' = epoch milliseconds,
            'iso' = ISO8601. The default depends on the `orient`. For
            ``orient='table'``, the default is 'iso'. For all other orients,
            the default is 'epoch'.
        double_precision : int, default 10
            The number of decimal places to use when encoding
            floating point values.
        force_ascii : bool, default True
            Force encoded string to be ASCII.
        date_unit : string, default 'ms' (milliseconds)
            The time unit to encode to, governs timestamp and ISO8601
            precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
            microsecond, and nanosecond respectively.
        default_handler : callable, default None
            Handler to call if object cannot otherwise be converted to a
            suitable format for JSON. Should receive a single argument which is
            the object to convert and return a serialisable object.
        lines : bool, default False
            If 'orient' is 'records' write out line delimited json format. Will
            throw ValueError if incorrect 'orient' since others are not list
            like.
        compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
            A string representing the compression to use in the output file,
            only used when the first argument is a filename. By default, the
            compression is inferred from the filename.
        index : bool, default True
            Whether to include the index values in the JSON string. Not
            including the index (``index=False``) is only supported when
            orient is 'split' or 'table'.
        """
        import cudf.io.json as json
        json.to_json(
            self,
            path_or_buf=path_or_buf,
            *args,
            **kwargs
        )

    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """
        Write the contained data to an HDF5 file using HDFStore.

        Hierarchical Data Format (HDF) is self-describing, allowing an
        application to interpret the structure and contents of a file with
        no outside information. One HDF file can hold a mix of related objects
        which can be accessed as a group or as individual objects.

        In order to add another DataFrame or Series to an existing HDF file
        please use append mode and a different a key.

        For more information see the :ref:`user guide
        <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables>`_.

        Parameters
        ----------
        path_or_buf : str or pandas.HDFStore
            File path or HDFStore object.
        key : str
            Identifier for the group in the store.
        mode : {'a', 'w', 'r+'}, default 'a'
            Mode to open file:
            - 'w': write, a new file is created (an existing file with
                the same name would be deleted).
            - 'a': append, an existing file is opened for reading and
                writing, and if the file does not exist it is created.
            - 'r+': similar to 'a', but the file must already exist.
        format : {'fixed', 'table'}, default 'fixed'
            Possible values:
            - 'fixed': Fixed format. Fast writing/reading. Not-appendable,
                nor searchable.
            - 'table': Table format. Write as a PyTables Table structure
                which may perform worse but allow more flexible operations
                like searching / selecting subsets of the data.
        append : bool, default False
            For Table formats, append the input data to the existing.
        data_columns :  list of columns or True, optional
            List of columns to create as indexed data columns for on-disk
            queries, or True to use all columns. By default only the axes
            of the object are indexed. `See Query via Data Columns
            <https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#hdf5-pytables>`_.
            Applicable only to format='table'.
        complevel : {0-9}, optional
            Specifies a compression level for data.
            A value of 0 disables compression.
        complib : {'zlib', 'lzo', 'bzip2', 'blosc'}, default 'zlib'
            Specifies the compression library to be used.
            As of v0.20.2 these additional compressors for Blosc are supported
            (default if no compressor specified: 'blosc:blosclz'):
            {'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy',
            'blosc:zlib', 'blosc:zstd'}.
            Specifying a compression library which is not available issues
            a ValueError.
        fletcher32 : bool, default False
            If applying compression use the fletcher32 checksum.
        dropna : bool, default False
            If true, ALL nan rows will not be written to store.
        errors : str, default 'strict'
            Specifies how encoding and decoding errors are to be handled.
            See the errors argument for :func:`open` for a full list
            of options.
        """
        import cudf.io.hdf as hdf
        hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack
        return dlpack.to_dlpack(self)

    def rename(self, index=None, copy=True):
        """
        Alter Series name.

        Change Series.name with a scalar value.

        Parameters
        ----------
        index : Scalar, optional
            Scalar to alter the Series.name attribute
        copy : boolean, default True
            Also copy underlying data

        Returns
        -------
        Series

        Difference from pandas:
          * Supports scalar values only for changing name attribute
          * Not supporting: inplace, level
        """
        out = self.copy(deep=False)
        out = out.set_index(self.index)
        if index:
            out.name = index

        return out.copy(deep=copy)


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
        if isinstance(arg, tuple):
            arg = list(arg)
        return self._sr[arg]

    def __setitem__(self, key, value):
        # throws an exception while updating
        msg = "updating columns using iloc is not allowed"
        raise ValueError(msg)
