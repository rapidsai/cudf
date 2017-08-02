from __future__ import print_function, division

from collections import OrderedDict

import numpy as np
import pandas as pd

from numba import cuda

from . import cudautils, utils, _gdf, formatting
from .buffer import Buffer
from .index import Index, RangeIndex, GenericIndex
from .settings import NOTSET, settings

class Series(object):
    """
    Data and null-masks.

    ``Series`` objects are used as columns of ``DataFrame``.
    """
    class Init(object):
        """
        Initializer object
        """
        def __init__(self, **kwargs):
            self._params = kwargs

        def parameters(self, **kwargs):
            dupset = frozenset(kwargs.keys()) & frozenset(self._params.keys())
            if dupset:
                raise ValueError("duplicated kws: {}",
                                 ', '.join(map(str, dupset)))
            kwargs.update(self._params)
            return kwargs

    @classmethod
    def from_any(cls, arbitrary):
        """Create Series from an arbitrary object

        Currently support inputs are:

        * ``Series``
        * ``Buffer``
        * numba device array
        * numpy array
        * pandas.Categorical
        """
        if isinstance(arbitrary, Series):
            return arbitrary

        # Handle pandas type
        if isinstance(arbitrary, pd.Categorical):
            return cls.from_categorical(arbitrary)

        # Handle internal types
        if isinstance(arbitrary, Buffer):
            return cls.from_buffer(arbitrary)
        elif cuda.devicearray.is_cuda_ndarray(arbitrary):
            return cls.from_array(arbitrary)
        else:
            if not isinstance(arbitrary, np.ndarray):
                arbitrary = np.asarray(arbitrary)
            return cls.from_array(arbitrary)

    @classmethod
    def from_categorical(cls, categorical, codes=None):
        """Creates from a pandas.Categorical

        If ``codes`` is defined, use it instead of ``categorical.codes``
        """
        from .categorical import CategoricalSeriesImpl

        # TODO fix mutability issue in numba to avoid the .copy()
        codes = (categorical.codes.copy()
                 if codes is None else codes)
        dtype = categorical.dtype
        # TODO pending pandas to be improved
        #       https://github.com/pandas-dev/pandas/issues/14711
        #       https://github.com/pandas-dev/pandas/pull/16015
        impl = CategoricalSeriesImpl(dtype, codes.dtype,
                                     categorical.categories,
                                     categorical.ordered)

        valid_codes = codes != -1
        buf = Buffer(codes)
        params = dict(buffer=buf, impl=impl)
        if not np.all(valid_codes):
            mask = utils.boolmask_to_bitmask(valid_codes)
            nnz = np.count_nonzero(valid_codes)
            null_count = codes.size - nnz
            params.update(dict(mask=Buffer(mask), null_count=null_count))
        return Series(cls.Init(**params))

    @classmethod
    def from_buffer(cls, buffer):
        """Create a Series from a ``Buffer``
        """
        return cls(cls.Init(buffer=buffer))

    @classmethod
    def from_array(cls, array):
        """Create a Series from an array-like object.
        """
        return cls.from_buffer(Buffer(array))

    @classmethod
    def from_masked_array(cls, data, mask, null_count=None):
        """Create a Series with null-mask.
        This is equivalent to:

            Series.from_any(data).set_mask(mask, null_count=null_count)

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
        return cls.from_any(data).set_mask(mask, null_count=null_count)

    def __new__(cls, arg, **kwargs):
        if isinstance(arg, cls.Init):
            instance = object.__new__(cls)
            instance._init_detail(**arg.parameters(**kwargs))
        else:
            instance = cls.from_any(arg, **kwargs)
        return instance

    def _init_detail(self, buffer=None, mask=None, null_count=None,
                     index=None, impl=None):
        """
        Actual initializer of the instance
        """
        from . import series_impl

        if index is not None and not isinstance(index, Index):
            raise TypeError('index not a Index type: got {!r}'.format(index))

        # Forces Series content to be contiguous
        if not buffer.is_contiguous():
            buffer = buffer.as_contiguous()

        self._size = buffer.size if buffer else 0
        self._data = buffer
        self._mask = mask
        self._index = RangeIndex(self._size) if index is None else index
        self._impl = (series_impl.get_default_impl(buffer.dtype)
                      if impl is None else impl)
        if null_count is None:
            if self._mask is not None:
                nnz = cudautils.count_nonzero_mask(self._mask.mem)
                null_count = self._size - nnz
            else:
                null_count = 0
        self._null_count = null_count
        self._cffi_view = _gdf.columnview(size=self._size, data=self._data,
                                          mask=self._mask)

    def _copy_construct_defaults(self):
        return dict(
            buffer=self._data,
            mask=self._mask,
            null_count=self._null_count,
            index=self._index,
            impl=self._impl,
        )

    def _copy_construct(self, **kwargs):
        """Shallow copy this object by replacing certain ctor args.
        """
        params = self._copy_construct_defaults()
        cls = type(self)
        params.update(kwargs)
        return cls(cls.Init(**params))

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
        return self._impl.as_index(self)

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
        if not isinstance(mask, Buffer):
            mask = Buffer(mask)
        if mask.dtype not in (np.dtype(np.uint8), np.dtype(np.int8)):
            msg = 'mask must be of byte; but got {}'.format(mask.dtype)
            raise ValueError(msg)
        return self._copy_construct(mask=mask, null_count=null_count)

    def __len__(self):
        """Returns the size of the ``Series`` including null values.
        """
        return self._size

    def __getitem__(self, arg):
        from . import series_impl

        if isinstance(arg, Series):
            return series_impl.select_by_boolmask(self, arg)

        elif isinstance(arg, slice):
            # compute mask slice
            start, stop = utils.normalize_slice(arg, len(self))
            if self.null_count > 0:
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError(arg)

                # compute new mask
                mask = self._get_mask_as_series()
                # slicing
                subdata = self._data.mem[arg]
                submask = mask[arg].as_mask()
                index = self.index[arg]
                return self._copy_construct(buffer=Buffer(subdata),
                                            mask=Buffer(submask),
                                            null_count=None,
                                            index=index)
            else:
                index = self.index[arg]
                newbuffer = self._data[arg]
                return self._copy_construct(buffer=newbuffer,
                                            mask=None,
                                            null_count=None,
                                            index=index)
        elif isinstance(arg, int):
            # The following triggers a IndexError if out-of-bound
            return self._impl.element_indexing(self, arg)
        else:
            raise NotImplementedError(type(arg))

    def take(self, indices):
        """Return Series by taking values from the corresponding *indices*.
        """
        indices = Buffer(indices).to_gpu_array()
        data = cudautils.gather(data=self.data.to_gpu_array(), index=indices)

        if self._mask:
            mask = self._get_mask_as_series().take(indices).as_mask()
            mask = Buffer(mask)
        else:
            mask = None
        index = self.index.take(indices)
        return self._copy_construct(buffer=Buffer(data), index=index,
                                    mask=mask)

    def _get_mask_as_series(self):
        mask = Series(cudautils.ones(len(self), dtype=np.bool))
        return mask.set_mask(self._mask).fillna(False)

    def __bool__(self):
        """Always raise TypeError when converting a Series
        into a boolean.
        """
        raise TypeError("can't compute boolean for {!r}".format(type(self)))

    def values_to_string(self, nrows=None):
        """Returns a list of string for each element.
        """
        values = self[:nrows]
        out = ['' if v is None else self._element_to_str(v) for v in values]
        return out

    def _element_to_str(self, value):
        return self._impl.element_to_str(value)

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
        if not isinstance(other, Series):
            return NotImplemented
        return self._impl.binary_operator(fn, self, other)

    def _unaryop(self, fn):
        """
        Internal util to call a unary operator *fn* on operands *self*.
        Return the output Series.  The output dtype is determined by the input
        operand.
        """
        return self._impl.unary_operator(fn, self)

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

    def _normalize_compare_value(self, other):
        if isinstance(other, Series):
            return other
        else:
            return self._impl.normalize_compare_value(self, other)

    def _unordered_compare(self, other, cmpops):
        other = self._normalize_compare_value(other)
        return self._impl.unordered_compare(cmpops, self, other)

    def _ordered_compare(self, other, cmpops):
        other = self._normalize_compare_value(other)
        return self._impl.ordered_compare(cmpops, self, other)

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
        return self._impl.cat(self)

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._impl.dtype

    def append(self, arbitrary):
        """Append values from another ``Series`` or array-like object.
        Returns a new copy.
        """
        other = Series.from_any(arbitrary)
        newsize = len(self) + len(other)
        # allocate memory
        mem = cuda.device_array(shape=newsize, dtype=self.data.dtype)
        newbuf = Buffer.from_empty(mem)
        # copy into new memory
        for buf in [self._data, other._data]:
            newbuf.extend(buf.to_gpu_array())
        # return new series
        return self.from_any(newbuf)

    @property
    def valid_count(self):
        """Number of non-null values"""
        return self._size - self._null_count

    @property
    def null_count(self):
        """Number of null values"""
        return self._null_count

    @property
    def has_null_mask(self):
        """A boolean indicating whether a null-mask is needed"""
        return self._mask is not None

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.has_null_mask:
            return self
        out = cudautils.fillna(data=self._data.to_gpu_array(),
                               mask=self._mask.to_gpu_array(),
                               value=value)
        return self.from_array(out)

    def to_dense_buffer(self, fillna=None):
        """Get dense (no null values) ``Buffer`` of the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        if fillna not in {None, 'pandas'}:
            raise ValueError('invalid for fillna')

        if self.has_null_mask:
            if fillna == 'pandas':
                # cast non-float types to float64
                col = (self.astype(np.float64)
                       if self.dtype.kind != 'f'
                       else self)
                # fill nan
                return col.fillna(np.nan)
            else:
                return self._copy_to_dense_buffer()
        else:
            return self._data

    def _copy_to_dense_buffer(self):
        data = self._data.to_gpu_array()
        mask = self._mask.to_gpu_array()
        nnz, mem = cudautils.copy_to_dense(data=data, mask=mask)
        return Buffer(mem, size=nnz, capacity=mem.size)

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
        return self.to_dense_buffer(fillna=fillna).to_array()

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
        return self.to_dense_buffer(fillna=fillna).to_gpu_array()

    def to_pandas(self):
        return self._impl.to_pandas(self)

    @property
    def data(self):
        """The gpu buffer for the data
        """
        return self._data

    @property
    def index(self):
        """The index object
        """
        return self._index

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        if self.has_null_mask:
            return self._mask
        else:
            raise ValueError('Series has no null mask')

    def as_mask(self):
        """Convert booleans to bitmask
        """
        # FIXME: inefficient
        return Buffer(utils.boolmask_to_bitmask(self.to_array())).to_gpu_array()

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
        return self._copy_construct(buffer=self.data.astype(dtype),
                                    impl=None)

    def argsort(self, ascending=True):
        """Returns a Series of int64 index that will sort the series.

        Uses stable parallel radixsort.

        Returns
        -------
        result: Series
        """
        return self._sort(ascending=ascending)[1]

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
        return vals.set_index(GenericIndex(inds.to_gpu_array()))

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
        return self._impl.sort_by_values(self, ascending=ascending)

    def reverse(self):
        """Reverse the Series
        """
        data = cudautils.reverse_array(self.to_gpu_array())
        index = GenericIndex(cudautils.reverse_array(self.index.gpu_values))
        return self._copy_construct(buffer=Buffer(data), index=index)


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
            buf = cudautils.apply_equal_constant(arr=self.to_gpu_array(),
                                                 val=cat, dtype=dtype)
            out.append(Series.from_array(buf))
        return out

    # Find / Search

    def find_first_value(self, value):
        """
        Returns offset of first value that matches
        """
        # FIXME: Inefficient find in CPU code
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not indices:
            raise ValueError('value not found')
        return indices[0, 0]

    def find_last_value(self, value):
        """
        Returns offset of last value that matches
        """
        arr = self.to_array()
        indices = np.argwhere(arr == value)
        if not indices:
            raise ValueError('value not found')
        return indices[-1, 0]

    #
    # Stats
    #
    def count(self):
        """The number of non-null values"""
        return self.valid_count

    def min(self):
        """Compute the min of the series
        """
        return self._impl.stats(self).min()

    def max(self):
        """Compute the max of the series
        """
        return self._impl.stats(self).max()

    def sum(self):
        """Compute the sum of the series"""
        return self._impl.stats(self).sum()

    def mean(self):
        """Compute the mean of the series
        """
        return self._impl.stats(self).mean()

    def std(self):
        """Compute the standard deviation of the series
        """
        return np.sqrt(self.var())

    def var(self):
        """Compute the variance of the series
        """
        mu, var = self.mean_var()
        return var

    def mean_var(self):
        """Compute mean and variance at the same time.
        """
        mu, var = self._impl.stats(self).mean_var()
        return mu, var

    def unique_k(self, k):
        """Returns a list of at most k unique values.
        """
        if self.null_count == len(self):
            return np.empty(0, dtype=self.dtype)
        arr = self.to_dense_buffer().to_gpu_array()
        return cudautils.compute_unique_k(arr, k=k)

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
        return Series.from_array(scaled)

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



