# Copyright (c) 2019, NVIDIA CORPORATION.

import pandas as pd
import numpy as np
import pyarrow as pa
import nvstrings
from numbers import Number
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import warnings

from cudf.dataframe import columnops, numerical, series
from cudf.dataframe.buffer import Buffer
from cudf.utils import utils, cudautils

import cudf.bindings.binops as cpp_binops
from cudf.bindings.cudf_cpp import get_ctype_ptr
from cudf.bindings.nvtx import nvtx_range_push, nvtx_range_pop
from librmm_cffi import librmm as rmm


class StringMethods(object):
    """
    This mimicks pandas `df.str` interface.
    """
    def __init__(self, parent, index=None):
        self._parent = parent
        self._index = index

    def __getattr__(self, attr, *args, **kwargs):
        from cudf.dataframe.series import Series
        if hasattr(self._parent._data, attr):
            passed_attr = getattr(self._parent._data, attr)
            if callable(passed_attr):
                def wrapper(*args, **kwargs):
                    return getattr(self._parent._data, attr)(*args, **kwargs)
                if isinstance(wrapper, nvstrings.nvstrings):
                    wrapper = Series(
                        columnops.as_column(wrapper),
                        index=self._index
                    )
                return wrapper
            else:
                return passed_attr
        else:
            raise AttributeError(attr)

    def len(self):
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
          Series or Index of int: A Series or Index of integer values
            indicating the length of each element in the Series or Index.
        """
        from cudf.dataframe.series import Series
        out_dev_arr = rmm.device_array(len(self._parent), dtype='int32')
        ptr = get_ctype_ptr(out_dev_arr)
        self._parent.data.len(ptr)

        mask = None
        if self._parent.null_count > 0:
            mask = self._parent.mask

        column = columnops.build_column(
            Buffer(out_dev_arr),
            np.dtype('int32'),
            mask=mask
        )
        return Series(column, index=self._index)

    def cat(self, others=None, sep=None, na_rep=None):
        """
        Concatenate strings in the Series/Index with given separator.

        If *others* is specified, this function concatenates the Series/Index
        and elements of others element-wise. If others is not passed, then all
        values in the Series/Index are concatenated into a single string with
        a given sep.

        Parameters
        ----------
            others : Series or List of str
                Strings to be appended.
                The number of strings must match size() of this instance.
                This must be either a Series of string dtype or a Python
                list of strings.

            sep : str
                If specified, this separator will be appended to each string
                before appending the others.

            na_rep : str
                This character will take the place of any null strings
                (not empty strings) in either list.

                - If `na_rep` is None, and `others` is None, missing values in
                the Series/Index are omitted from the result.
                - If `na_rep` is None, and `others` is not None, a row
                containing a missing value in any of the columns (before
                concatenation) will have a missing value in the result.

        Returns
        -------
        concat : str or Series/Index of str dtype
            If `others` is None, `str` is returned, otherwise a `Series/Index`
            (same type as caller) of str dtype is returned.
        """
        from cudf.dataframe import Series, Index
        if isinstance(others, (Series, Index)):
            assert others.dtype == np.dtype('object')
            others = others.data
        out = Series(
            self._parent.data.cat(others=others, sep=sep, na_rep=na_rep),
            index=self._index
        )
        if len(out) == 1 and others is None:
            out = out[0]
        return out

    def join(self, sep):
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.
        """
        raise NotImplementedError("Columns of arrays / lists are not yet "
                                  "supported")

    def extract(self, pat, flags=0, expand=True):
        """
        Extract capture groups in the regex `pat` as columns in a DataFrame.

        For each subject string in the Series, extract groups from the first
        match of regular expression `pat`.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        expand : bool, default True
            If True, return DataFrame with on column per capture group.
            If False, return a Series/Index if there is one capture group or
            DataFrame if there are multiple capture groups.

        Returns
        -------
        DataFrame or Series/Index
            A DataFrame with one row for each subject string, and one column
            for each group. If `expand=False` and `pat` has only one capture
            group, then return a Series/Index.

        Notes
        -----
        The `flags` parameter is not yet supported and will raise a
        NotImplementedError if anything other than the default value is passed.
        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        from cudf.dataframe import DataFrame, Series
        out = self._parent.data.extract(pat)
        if len(out) == 1 and expand is False:
            return Series(
                out[0],
                index=self._index
            )
        else:
            out_df = DataFrame(index=self._index)
            for idx, val in enumerate(out):
                out_df[idx] = val
            return out_df

    def contains(self, pat, case=True, flags=0, na=np.nan, regex=True):
        """
        Test if pattern or regex is contained within a string of a Series or
        Index.

        Return boolean Series or Index based on whether a given pattern or
        regex is contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        regex : bool, default True
            If True, assumes the pattern is a regular expression.
            If False, treats the pattern as a literal string.

        Returns
        -------
        Series/Index of bool dtype
            A Series/Index of boolean dtype indicating whether the given
            pattern is contained within the string of each element of the
            Series/Index.

        Notes
        -----
        The parameters `case`, `flags`, and `na` are not yet supported and
        will raise a NotImplementedError if anything other than the default
        value is set.
        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        elif flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")
        elif na is not np.nan:
            raise NotImplementedError("`na` parameter is not yet supported")

        from cudf.dataframe import Series
        out_dev_arr = rmm.device_array(len(self._parent), dtype='bool')
        ptr = get_ctype_ptr(out_dev_arr)
        self._parent.data.contains(pat, regex=regex, devptr=ptr)

        mask = None
        if self._parent.null_count > 0:
            mask = self._parent.mask

        column = columnops.build_column(
            Buffer(out_dev_arr),
            np.dtype('bool'),
            mask=mask
        )

        return Series(
            column,
            index=self._index
        )

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True):
        """
        Replace occurences of pattern/regex in the Series/Index with some other
        string.

        Parameters
        ----------
        pat : str
            String to be replaced as a character sequence or regular
            expression.
        repl : str
            String to be used as replacement.
        n : int, default -1 (all)
            Number of replacements to make from the start.
        regex : bool, default True
            If True, assumes the pattern is a regular expression.
            If False, treats the pattern as a literal string.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all matching occurrences of pat replaced
            by repl.

        Notes
        -----
        The parameters `case` and `flags` are not yet supported and will raise
        a NotImplementedError if anything other than the default value is set.
        """
        if case is not None:
            raise NotImplementedError("`case` parameter is not yet supported")
        elif flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        from cudf.dataframe import Series
        return Series(
            self._parent.data.replace(pat, repl, n=n, regex=regex),
            index=self._index
        )

    def lower(self):
        """
        Convert strings in the Series/Index to lowercase.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all strings converted to lowercase.
        """
        from cudf.dataframe import Series
        return Series(
            self._parent.data.lower(),
            index=self._index
        )

    def split(self, pat=None, n=-1, expand=True):
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the beginning, at the
        specified delimiter string.

        Parameters
        ----------
        pat : str, default ' ' (space)
            String to split on, does not yet support regular expressions.
        n : int, default -1 (all)
            Limit number of splits in output. `None`, 0, and -1 will all be
            interpreted as "all splits".

        Returns
        -------
        DataFrame
            Returns a DataFrame with each split as a column.

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        NotImplementedError if anything other than the default value is set.
        """
        if expand is not True:
            raise NotImplementedError("`expand` parameter is not supported")

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        from cudf.dataframe import DataFrame
        out_df = DataFrame(index=self._index)
        out = self._parent.data.split(delimiter=pat, n=n)
        for idx, val in enumerate(out):
            out_df[idx] = val
        return out_df


class StringColumn(columnops.TypedColumnBase):
    """Implements operations for Columns of String type
    """
    def __init__(self, data, null_count=None, **kwargs):
        """
        Parameters
        ----------
        data : nvstrings.nvstrings
            The nvstrings object
        null_count : int; optional
            The number of null values in the mask.
        """
        from collections.abc import Sequence
        if isinstance(data, Sequence):
            data = nvstrings.to_device(data)
        assert isinstance(data, nvstrings.nvstrings)
        self._data = data
        self._dtype = np.dtype("object")

        if null_count is None:
            null_count = data.null_count()
        self._null_count = null_count
        self._mask = None
        if self._null_count > 0:
            mask_size = utils.calc_chunk_size(len(self.data),
                                              utils.mask_bitsize)
            out_mask_arr = rmm.device_array(mask_size, dtype='int8')
            out_mask_ptr = get_ctype_ptr(out_mask_arr)
            self.data.set_null_bitmask(out_mask_ptr, bdevmem=True)
            self._mask = Buffer(out_mask_arr)
        self._nvcategory = None
        self._indices = None

    def __reduce__(self):
        cpumem = self.to_arrow()
        return columnops.as_column, (cpumem, False, np.dtype('object'))

    def str(self, index=None):
        return StringMethods(self, index=index)

    def __len__(self):
        return self._data.size()

    @property
    def dtype(self):
        return self._dtype

    @property
    def data(self):
        """ nvstrings object """
        return self._data

    @property
    def null_count(self):
        return self._null_count

    @property
    def mask(self):
        """Validity mask buffer
        """
        return self._mask

    @property
    def nvcategory(self):
        if self._nvcategory is None:
            import nvcategory as nvc
            self._nvcategory = nvc.from_strings(self.data)
        return self._nvcategory

    @property
    def indices(self):
        if self._indices is None:
            out_dev_arr = rmm.device_array(
                self.nvcategory.size(),
                dtype='int32'
            )
            ptr = get_ctype_ptr(out_dev_arr)
            self.nvcategory.values(devptr=ptr)
            self._indices = Buffer(out_dev_arr)
        return self._indices

    def element_indexing(self, arg):
        if isinstance(arg, Number):
            arg = int(arg)
            if arg > (len(self) - 1) or arg < 0:
                raise IndexError
            out = self._data[arg]
        elif isinstance(arg, slice):
            out = self._data[arg]
        elif isinstance(arg, list):
            out = self._data[arg]
        elif isinstance(arg, np.ndarray):
            gpu_arr = rmm.to_device(arg)
            return self.element_indexing(gpu_arr)
        elif isinstance(arg, DeviceNDArray):
            # NVStrings gather call expects an array of int32s
            arg = cudautils.astype(arg, np.dtype('int32'))
            if len(arg) > 0:
                gpu_ptr = get_ctype_ptr(arg)
                out = self._data.gather(gpu_ptr, len(arg))
            else:
                out = self._data.gather([])
        else:
            raise NotImplementedError(type(arg))

        if len(out) == 1:
            return out.to_host()[0]
        else:
            return columnops.as_column(out)

    def __getitem__(self, arg):
        return self.element_indexing(arg)

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        elif dtype in (np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                       np.dtype('int64')):
            out_arr = rmm.device_array(shape=len(self), dtype='int32')
            out_ptr = get_ctype_ptr(out_arr)
            self.str().stoi(devptr=out_ptr)
        elif dtype in (np.dtype('float32'), np.dtype('float64')):
            out_arr = rmm.device_array(shape=len(self), dtype='float32')
            out_ptr = get_ctype_ptr(out_arr)
            self.str().stof(devptr=out_ptr)
        out_col = columnops.as_column(out_arr)
        return out_col.astype(dtype)

    def to_arrow(self):
        sbuf = np.empty(self._data.byte_count(), dtype='int8')
        obuf = np.empty(len(self._data) + 1, dtype='int32')

        mask_size = utils.calc_chunk_size(len(self._data), utils.mask_bitsize)
        nbuf = np.empty(mask_size, dtype='int8')

        self.str().to_offsets(sbuf, obuf, nbuf=nbuf)
        sbuf = pa.py_buffer(sbuf)
        obuf = pa.py_buffer(obuf)
        nbuf = pa.py_buffer(nbuf)
        if self.null_count == len(self):
            return pa.NullArray.from_buffers(
                pa.null(), len(self), np.empty(0), self.null_count
            )
        else:
            return pa.StringArray.from_buffers(
                len(self._data), obuf, sbuf, nbuf, self._data.null_count()
            )

    def to_pandas(self, index=None):
        pd_series = self.to_arrow().to_pandas()
        return pd.Series(pd_series, index=index)

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.

        Raises
        ------
        ``NotImplementedError`` if there are nulls
        """
        if fillna is not None:
            warnings.warn("fillna parameter not supported for string arrays")

        return self.to_arrow().to_pandas()

    def sort_by_values(self, ascending=True, na_position="last"):
        if na_position == "last":
            nullfirst = False
        elif na_position == "first":
            nullfirst = True

        idx_dev_arr = rmm.device_array(len(self), dtype='int32')
        dev_ptr = get_ctype_ptr(idx_dev_arr)
        self.data.order(2, asc=ascending, nullfirst=nullfirst, devptr=dev_ptr)

        col_inds = numerical.NumericalColumn(
            data=Buffer(idx_dev_arr),
            mask=None,
            null_count=0,
            dtype=idx_dev_arr.dtype
        )

        col_keys = self[col_inds.data.mem]

        return col_keys, col_inds

    def _replace_defaults(self):
        params = {
            'data': self.data,
            'null_count': self.null_count,
        }
        return params

    def copy(self, deep=True):
        params = self._replace_defaults()
        return type(self)(**params)

    def unordered_compare(self, cmpop, rhs):
        return string_column_binop(self, rhs, op=cmpop)

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with * fill_value *
        """

        if not isinstance(fill_value, str) and \
            not(
                isinstance(fill_value, series.Series) and
                isinstance(fill_value._column, StringColumn)
                ):
            raise TypeError("fill_value must be a string or a string series")

        # replace fill_value with nvstrings
        # if it is a column

        if isinstance(fill_value, series.Series):
            if len(fill_value) < len(self):
                raise ValueError("fill value series must be of same or "
                                 "greater length than the series to be filled")

            fill_value = fill_value[: len(self)]._column._data

        filled_data = self._data.fillna(fill_value)
        result = StringColumn(filled_data)
        result = result.replace(mask=None)
        return self._mimic_inplace(result, inplace)


def string_column_binop(lhs, rhs, op):
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = lhs.has_null_mask or rhs.has_null_mask
    out = columnops.column_empty_like(lhs, dtype='bool', masked=masked)
    # Call and fix null_count
    null_count = cpp_binops.apply_op(lhs=lhs, rhs=rhs, out=out, op=op)

    result = out.replace(null_count=null_count)
    nvtx_range_pop()
    return result
