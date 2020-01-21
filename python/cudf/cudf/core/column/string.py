# Copyright (c) 2019, NVIDIA CORPORATION.

import functools
import pickle
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda

import nvstrings
import rmm

import cudf._lib as libcudf
from cudf._lib.nvtx import nvtx_range_pop, nvtx_range_push
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.utils import cudautils, utils
from cudf.utils.dtypes import is_list_like

_str_to_numeric_typecast_functions = {
    np.dtype("int32"): nvstrings.nvstrings.stoi,
    np.dtype("int64"): nvstrings.nvstrings.stol,
    np.dtype("float32"): nvstrings.nvstrings.stof,
    np.dtype("float64"): nvstrings.nvstrings.stod,
    np.dtype("bool"): nvstrings.nvstrings.to_booleans,
    # TODO: support Date32 UNIX days
    # np.dtype("datetime64[D]"): nvstrings.nvstrings.timestamp2int,
    np.dtype("datetime64[s]"): nvstrings.nvstrings.timestamp2int,
    np.dtype("datetime64[ms]"): nvstrings.nvstrings.timestamp2int,
    np.dtype("datetime64[us]"): nvstrings.nvstrings.timestamp2int,
    np.dtype("datetime64[ns]"): nvstrings.nvstrings.timestamp2int,
}

_numeric_to_str_typecast_functions = {
    np.dtype("int32"): nvstrings.itos,
    np.dtype("int64"): nvstrings.ltos,
    np.dtype("float32"): nvstrings.ftos,
    np.dtype("float64"): nvstrings.dtos,
    np.dtype("bool"): nvstrings.from_booleans,
    # TODO: support Date32 UNIX days
    # np.dtype("datetime64[D]"): nvstrings.int2timestamp,
    np.dtype("datetime64[s]"): nvstrings.int2timestamp,
    np.dtype("datetime64[ms]"): nvstrings.int2timestamp,
    np.dtype("datetime64[us]"): nvstrings.int2timestamp,
    np.dtype("datetime64[ns]"): nvstrings.int2timestamp,
}


class StringMethods(object):
    """
    This mimicks pandas `df.str` interface.
    """

    def __init__(self, parent, index=None, name=None):
        self._parent = parent
        self._index = index
        self._name = name

    def __getattr__(self, attr, *args, **kwargs):
        from cudf.core.series import Series

        if hasattr(self._parent.nvstrings, attr):
            passed_attr = getattr(self._parent.nvstrings, attr)
            if callable(passed_attr):

                @functools.wraps(passed_attr)
                def wrapper(*args, **kwargs):
                    ret = passed_attr(*args, **kwargs)
                    if isinstance(ret, nvstrings.nvstrings):
                        ret = Series(
                            column.as_column(ret),
                            index=self._index,
                            name=self._name,
                        )
                    return ret

                return wrapper
            else:
                return passed_attr
        else:
            raise AttributeError(attr)

    def __dir__(self):
        keys = dir(type(self))
        return set(keys + dir(self._parent.nvstrings))

    def len(self):
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
          Series or Index of int: A Series or Index of integer values
            indicating the length of each element in the Series or Index.
        """
        from cudf.core.series import Series

        out_dev_arr = rmm.device_array(len(self._parent), dtype="int32")
        ptr = libcudf.cudf.get_ctype_ptr(out_dev_arr)
        self._parent.nvstrings.len(ptr)

        mask = None
        if self._parent.has_nulls:
            mask = self._parent.mask

        col = column.build_column(
            Buffer(out_dev_arr), np.dtype("int32"), mask=mask
        )
        return Series(col, index=self._index, name=self._name)

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
        from cudf.core import Series, Index

        if isinstance(others, Series):
            assert others.dtype == np.dtype("object")
            others = others._column.nvstrings
        elif isinstance(others, Index):
            assert others.dtype == np.dtype("object")
            others = others.as_column().nvstrings
        elif isinstance(others, StringMethods):
            """
            If others is a StringMethods then
            raise an exception
            """
            msg = "series.str is an accessor, not an array-like of strings."
            raise ValueError(msg)
        elif is_list_like(others) and others:
            """
            If others is a list-like object (in our case lists & tuples)
            just another Series/Index, great go ahead with concatenation.
            """

            """
            Picking first element and checking if it really adheres to
            list like conditions, if not we switch to next case

            Note: We have made a call not to iterate over the entire list as
            it could be more expensive if it was of very large size.
            Thus only doing a sanity check on just the first element of list.
            """
            first = others[0]

            if is_list_like(first) or isinstance(
                first, (Series, Index, pd.Series, pd.Index)
            ):
                """
                Internal elements in others list should also be
                list-like and not a regular string/byte
                """
                first = None
                for frame in others:
                    if not isinstance(frame, Series):
                        """
                        Make sure all inputs to .cat function call
                        are of type nvstrings so creating a Series object.
                        """
                        frame = Series(frame, dtype="str")

                    if first is None:
                        """
                        extracting nvstrings pointer since
                        `frame` is of type Series/Index and
                        first isn't yet initialized.
                        """
                        first = frame._column.nvstrings
                    else:
                        assert frame.dtype == np.dtype("object")
                        frame = frame._column.nvstrings
                        first = first.cat(frame, sep=sep, na_rep=na_rep)

                others = first
            elif not is_list_like(first):
                """
                Picking first element and checking if it really adheres to
                non-list like conditions.

                Note: We have made a call not to iterate over the entire
                list as it could be more expensive if it was of very
                large size. Thus only doing a sanity check on just the
                first element of list.
                """
                others = Series(others)
                others = others._column.nvstrings
        elif isinstance(others, (pd.Series, pd.Index)):
            others = Series(others)
            others = others._column.nvstrings

        data = self._parent.nvstrings.cat(
            others=others, sep=sep, na_rep=na_rep
        )
        out = Series(data, index=self._index, name=self._name)
        if len(out) == 1 and others is None:
            out = out[0]
        return out

    def join(self, sep):
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.
        """
        raise NotImplementedError(
            "Columns of arrays / lists are not yet " "supported"
        )

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

        from cudf.core import DataFrame, Series

        out = self._parent.nvstrings.extract(pat)
        if len(out) == 1 and expand is False:
            return Series(out[0], index=self._index, name=self._name)
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

        from cudf.core import Series

        out_dev_arr = rmm.device_array(len(self._parent), dtype="bool")
        ptr = libcudf.cudf.get_ctype_ptr(out_dev_arr)
        self._parent.nvstrings.contains(pat, regex=regex, devptr=ptr)

        mask = None
        if self._parent.has_nulls:
            mask = self._parent.mask

        col = column.build_column(
            Buffer(out_dev_arr), dtype=np.dtype("bool"), mask=mask
        )

        return Series(col, index=self._index, name=self._name)

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

        from cudf.core import Series

        return Series(
            self._parent.nvstrings.replace(pat, repl, n=n, regex=regex),
            index=self._index,
            name=self._name,
        )

    def lower(self):
        """
        Convert strings in the Series/Index to lowercase.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all strings converted to lowercase.
        """
        from cudf.core import Series

        return Series(
            self._parent.nvstrings.lower(), index=self._index, name=self._name
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

        from cudf.core import DataFrame

        out_df = DataFrame(index=self._index)
        out = self._parent.nvstrings.split(delimiter=pat, n=n)

        for idx, val in enumerate(out):
            out_df[idx] = val
        return out_df


class StringColumn(column.ColumnBase):
    """Implements operations for Columns of String type
    """

    def __init__(self, mask=None, offset=0, children=()):
        """
        Parameters
        ----------
        mask : Buffer
            The validity mask
        offset : int
            Data offset
        children : Tuple[Column]
            Two non-null columns containing the string data and offsets
            respectively
        """

        data = Buffer.empty(0)
        dtype = np.dtype("object")

        if children[0].size == 0:
            size = 0
        else:
            # one less because the last element of offsets is the number of
            # bytes in the data buffer
            size = children[0].size - 1

        super().__init__(data, size, dtype, mask=mask, children=children)

        self._nvstrings = None
        self._nvcategory = None
        self._indices = None

    def __contains__(self, item):
        return True in self.str().contains(f"^{item}$")._column

    def __reduce__(self):
        cpumem = self.to_arrow()
        return column.as_column, (cpumem, False, np.dtype("object"))

    def str(self, index=None, name=None):
        return StringMethods(self, index=index, name=name)

    def __sizeof__(self):
        n = self.nvstrings.device_memory()
        if self.mask:
            n += self.mask.size
        return n

    def _memory_usage(self, deep=False):
        if deep:
            return self.__sizeof__()
        else:
            return self.str().size() * self.dtype.itemsize

    def __len__(self):
        return self.nvstrings.size()

    @property
    def nvstrings(self):
        if self._nvstrings is None:
            if self.nullable:
                mask_ptr = self.mask.ptr
            else:
                mask_ptr = None
            if self.size == 0:
                self._nvstrings = nvstrings.to_device([])
            else:
                self._nvstrings = nvstrings.from_offsets(
                    self.children[1].data.ptr,
                    self.children[0].data.ptr,
                    self.size,
                    mask_ptr,
                    ncount=self.null_count,
                    bdevmem=True,
                )
        return self._nvstrings

    @property
    def nvcategory(self):
        if self._nvcategory is None:
            import nvcategory as nvc

            self._nvcategory = nvc.from_strings(self.nvstrings)
        return self._nvcategory

    @nvcategory.setter
    def nvcategory(self, nvc):
        self._nvcategory = nvc

    def _set_mask(self, value):
        self._nvstrings = None
        self._nvcategory = None
        super()._set_mask(value)

    @property
    def indices(self):
        if self._indices is None:
            out_dev_arr = rmm.device_array(
                self.nvcategory.size(), dtype="int32"
            )
            ptr = libcudf.cudf.get_ctype_ptr(out_dev_arr)
            self.nvcategory.values(devptr=ptr)
            self._indices = out_dev_arr
        return self._indices

    def as_numerical_column(self, dtype, **kwargs):

        mem_dtype = np.dtype(dtype)
        str_dtype = mem_dtype
        out_dtype = mem_dtype

        if mem_dtype.type in (np.int8, np.int16):
            mem_dtype = np.dtype(np.int32)
            str_dtype = mem_dtype
        elif mem_dtype.type is np.datetime64:
            kwargs.update(units=np.datetime_data(mem_dtype)[0])
            mem_dtype = np.dtype(np.int64)
            if "format" not in kwargs:
                if len(self.nvstrings) > 0:
                    # infer on host from the first not na element
                    fmt = pd.core.tools.datetimes._guess_datetime_format(
                        self[self.notna()][0]
                    )
                    kwargs.update(format=fmt)
            else:
                fmt = None

        out_arr = rmm.device_array(shape=len(self), dtype=mem_dtype)
        out_ptr = libcudf.cudf.get_ctype_ptr(out_arr)
        kwargs.update({"devptr": out_ptr})

        _str_to_numeric_typecast_functions[str_dtype](self.nvstrings, **kwargs)

        out_col = column.as_column(out_arr)

        if self.has_nulls:
            mask_size = utils.calc_chunk_size(
                len(self.nvstrings), utils.mask_bitsize
            )
            out_mask = column.column_empty(
                mask_size, dtype="int8", masked=False
            ).data
            out_mask_ptr = out_mask.ptr
            self.nvstrings.set_null_bitmask(out_mask_ptr, bdevmem=True)
            out_col.mask = out_mask

        return out_col.astype(out_dtype)

    def as_datetime_column(self, dtype, **kwargs):
        return self.as_numerical_column(dtype, **kwargs)

    def as_string_column(self, dtype, **kwargs):
        return self

    def to_arrow(self):
        sbuf = np.empty(self.nvstrings.byte_count(), dtype="int8")
        obuf = np.empty(len(self.nvstrings) + 1, dtype="int32")

        mask_size = utils.calc_chunk_size(
            len(self.nvstrings), utils.mask_bitsize
        )
        nbuf = np.empty(mask_size, dtype="int8")

        self.str().to_offsets(sbuf, obuf, nbuf=nbuf)
        sbuf = pa.py_buffer(sbuf)
        obuf = pa.py_buffer(obuf)
        nbuf = pa.py_buffer(nbuf)
        if self.null_count == len(self):
            return pa.NullArray.from_buffers(
                pa.null(), len(self), [pa.py_buffer((b""))], self.null_count
            )
        else:
            return pa.StringArray.from_buffers(
                len(self.nvstrings),
                obuf,
                sbuf,
                nbuf,
                self.nvstrings.null_count(),
            )

    def to_pandas(self, index=None):
        pd_series = self.to_arrow().to_pandas()
        if index is not None:
            pd_series.index = index
        return pd_series

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

        return self.to_arrow().to_pandas().array

    def serialize(self):
        header = {"null_count": self.null_count}
        header["type"] = pickle.dumps(type(self))
        frames = []
        sub_headers = []

        sbuf = rmm.device_array(self.nvstrings.byte_count(), dtype="int8")
        obuf = rmm.device_array(len(self.nvstrings) + 1, dtype="int32")
        mask_size = utils.calc_chunk_size(
            len(self.nvstrings), utils.mask_bitsize
        )
        nbuf = rmm.device_array(mask_size, dtype="int8")
        self.nvstrings.to_offsets(
            libcudf.cudf.get_ctype_ptr(sbuf),
            libcudf.cudf.get_ctype_ptr(obuf),
            nbuf=libcudf.cudf.get_ctype_ptr(nbuf),
            bdevmem=True,
        )
        for item in [nbuf, sbuf, obuf]:
            sheader = item.__cuda_array_interface__.copy()
            sheader["dtype"] = item.dtype.str
            sub_headers.append(sheader)
            frames.append(item)

        header["nvstrings"] = len(self.nvstrings)
        header["subheaders"] = sub_headers
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        # Deserialize the mask, value, and offset frames
        arrays = []

        for each_frame in frames:
            if hasattr(each_frame, "__cuda_array_interface__"):
                each_frame = cuda.as_cuda_array(each_frame)
            elif isinstance(each_frame, memoryview):
                each_frame = np.asarray(each_frame)
                each_frame = cudautils.to_device(each_frame)

            arrays.append(libcudf.cudf.get_ctype_ptr(each_frame))

        # Use from_offsets to get nvstring data.
        # Note: array items = [nbuf, sbuf, obuf]
        scount = header["nvstrings"]
        data = nvstrings.from_offsets(
            arrays[1],
            arrays[2],
            scount,
            nbuf=arrays[0],
            ncount=header["null_count"],
            bdevmem=True,
        )
        return column.as_column(data)

    def sort_by_values(self, ascending=True, na_position="last"):
        if na_position == "last":
            nullfirst = False
        elif na_position == "first":
            nullfirst = True

        idx_dev_arr = rmm.device_array(len(self), dtype="int32")
        dev_ptr = libcudf.cudf.get_ctype_ptr(idx_dev_arr)
        self.nvstrings.order(
            2, asc=ascending, nullfirst=nullfirst, devptr=dev_ptr
        )

        col_inds = column.build_column(
            Buffer(idx_dev_arr), idx_dev_arr.dtype, mask=None
        )

        col_keys = self[col_inds.data_array_view]

        return col_keys, col_inds

    def copy(self, deep=True):
        return column.as_column(self.nvstrings.copy())

    def unordered_compare(self, cmpop, rhs):
        return _string_column_binop(self, rhs, op=cmpop)

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*
        """
        to_replace = column.as_column(to_replace)
        replacement = column.as_column(replacement)
        if len(to_replace) == 1 and len(replacement) == 1:
            to_replace = to_replace.nvstrings.to_host()[0]
            replacement = replacement.nvstrings.to_host()[0]
            result = self.nvstrings.replace(to_replace, replacement)
            return column.as_column(result)
        else:
            raise NotImplementedError(
                "StringColumn currently only supports replacing"
                " single values"
            )

    def fillna(self, fill_value, inplace=False):
        """
        Fill null values with * fill_value *
        """
        from cudf.core.series import Series

        if not isinstance(fill_value, str) and not (
            isinstance(fill_value, Series)
            and isinstance(fill_value._column, StringColumn)
        ):
            raise TypeError("fill_value must be a string or a string series")

        # replace fill_value with nvstrings
        # if it is a column

        if isinstance(fill_value, Series):
            if len(fill_value) < len(self):
                raise ValueError(
                    "fill value series must be of same or "
                    "greater length than the series to be filled"
                )

            fill_value = fill_value[: len(self)]._column.nvstrings

        filled_data = self.nvstrings.fillna(fill_value)
        result = column.as_column(filled_data)
        result.mask = None
        return self._mimic_inplace(result, inplace)

    def _find_first_and_last(self, value):
        found_indices = self.str().contains(f"^{value}$")._column
        found_indices = libcudf.typecast.cast(found_indices, dtype=np.int32)
        first = column.as_column(found_indices).find_first_value(1)
        last = column.as_column(found_indices).find_last_value(1)
        return first, last

    def find_first_value(self, value, closest=False):
        return self._find_first_and_last(value)[0]

    def find_last_value(self, value, closest=False):
        return self._find_first_and_last(value)[1]

    def unique(self, method="sort"):
        """
        Get unique strings in the data
        """
        import nvcategory as nvc

        return column.as_column(nvc.from_strings(self.nvstrings).keys())

    def normalize_binop_value(self, other):
        if isinstance(other, column.Column):
            return other.astype(self.dtype)
        elif isinstance(other, str) or other is None:
            col = utils.scalar_broadcast_to(
                other, size=len(self), dtype="object"
            )
            return col
        else:
            raise TypeError("cannot broadcast {}".format(type(other)))

    def default_na_value(self):
        return None

    def binary_operator(self, binop, rhs, reflect=False):
        lhs = self
        if reflect:
            lhs, rhs = rhs, lhs
        if isinstance(rhs, StringColumn) and binop == "add":
            return lhs.nvstrings.cat(others=rhs.nvstrings)
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))

    @property
    def is_unique(self):
        return len(self.unique()) == len(self)

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_increasing = False
            else:
                self._is_monotonic_increasing = libcudf.issorted.issorted(
                    columns=[self]
                )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            if self.nullable and self.has_nulls:
                self._is_monotonic_decreasing = False
            else:
                self._is_monotonic_decreasing = libcudf.issorted.issorted(
                    columns=[self], descending=[1]
                )
        return self._is_monotonic_decreasing

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Strings are not yet supported via `__cuda_array_interface__`"
        )


def _string_column_binop(lhs, rhs, op):
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = lhs.nullable or rhs.nullable
    out = column.column_empty_like(lhs, dtype="bool", masked=masked)
    # Call and fix null_count
    _ = libcudf.binops.apply_op(lhs=lhs, rhs=rhs, out=out, op=op)
    nvtx_range_pop()
    return out
