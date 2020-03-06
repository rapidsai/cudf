# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import functools
import pickle
import warnings
from codecs import decode

import numpy as np
import pandas as pd
import pyarrow as pa

import nvstrings
import rmm

import cudf._lib as libcudf
import cudf._libxx as libcudfxx
import cudf._libxx.string_casting as str_cast
from cudf._lib.nvtx import nvtx_range_pop, nvtx_range_push
from cudf._libxx.strings.char_types import (
    is_alnum as cpp_is_alnum,
    is_alpha as cpp_is_alpha,
    is_decimal as cpp_is_decimal,
    is_digit as cpp_is_digit,
    is_numeric as cpp_is_numeric,
)
from cudf._libxx.strings.combine import (
    concatenate as cpp_concatenate,
    join as cpp_join,
)
from cudf._libxx.strings.padding import (
    PadSide,
    center as cpp_center,
    ljust as cpp_ljust,
    pad as cpp_pad,
    rjust as cpp_rjust,
    zfill as cpp_zfill,
)
from cudf._libxx.strings.replace import (
    insert as cpp_string_insert,
    slice_replace as cpp_slice_replace,
)
from cudf._libxx.strings.split.partition import (
    partition as cpp_partition,
    rpartition as cpp_rpartition,
)
from cudf._libxx.strings.split.split import (
    rsplit as cpp_rsplit,
    split as cpp_split,
)
from cudf._libxx.strings.strip import (
    lstrip as cpp_lstrip,
    rstrip as cpp_rstrip,
    strip as cpp_strip,
)
from cudf._libxx.strings.substring import slice_from as cpp_slice_from
from cudf.core.buffer import Buffer
from cudf.core.column import column
from cudf.utils import utils
from cudf.utils.dtypes import is_list_like, is_scalar

_str_to_numeric_typecast_functions = {
    np.dtype("int8"): str_cast.stoi8,
    np.dtype("int16"): str_cast.stoi16,
    np.dtype("int32"): str_cast.stoi,
    np.dtype("int64"): str_cast.stol,
    np.dtype("float32"): str_cast.stof,
    np.dtype("float64"): str_cast.stod,
    np.dtype("bool"): str_cast.to_booleans,
    # TODO: support Date32 UNIX days
    # np.dtype("datetime64[D]"): str_cast.timestamp2int,
    np.dtype("datetime64[s]"): str_cast.timestamp2int,
    np.dtype("datetime64[ms]"): str_cast.timestamp2int,
    np.dtype("datetime64[us]"): str_cast.timestamp2int,
    np.dtype("datetime64[ns]"): str_cast.timestamp2int,
}

_numeric_to_str_typecast_functions = {
    np.dtype("int8"): str_cast.i8tos,
    np.dtype("int16"): str_cast.i16tos,
    np.dtype("int32"): str_cast.itos,
    np.dtype("int64"): str_cast.ltos,
    np.dtype("float32"): str_cast.ftos,
    np.dtype("float64"): str_cast.dtos,
    np.dtype("bool"): str_cast.from_booleans,
    # TODO: support Date32 UNIX days
    # np.dtype("datetime64[D]"): str_cast.int2timestamp,
    np.dtype("datetime64[s]"): str_cast.int2timestamp,
    np.dtype("datetime64[ms]"): str_cast.int2timestamp,
    np.dtype("datetime64[us]"): str_cast.int2timestamp,
    np.dtype("datetime64[ns]"): str_cast.int2timestamp,
}


class StringMethods(object):
    """
    This mimicks pandas `df.str` interface.
    """

    def __init__(self, column, parent=None):
        self._column = column
        self._parent = parent

    def __getattr__(self, attr, *args, **kwargs):
        from cudf.core.series import Series

        # TODO: Remove when all needed string compute APIs are ported
        if hasattr(self._column.nvstrings, attr):
            passed_attr = getattr(self._column.nvstrings, attr)
            if callable(passed_attr):

                @functools.wraps(passed_attr)
                def wrapper(*args, **kwargs):
                    ret = passed_attr(*args, **kwargs)
                    if isinstance(ret, nvstrings.nvstrings):
                        ret = Series(
                            column.as_column(ret),
                            index=self._parent.index,
                            name=self._parent.name,
                        )
                    return ret

                return wrapper
            else:
                return passed_attr
        else:
            raise AttributeError(attr)

    def _return_or_inplace(self, new_col, **kwargs):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """
        from cudf import Series, DataFrame, MultiIndex
        from cudf.core.index import Index, as_index

        inplace = kwargs.get("inplace", False)

        if inplace:
            self._parent._mimic_inplace(new_col, inplace=True)
        else:
            expand = kwargs.get("expand", False)
            if expand or isinstance(self._parent, (DataFrame, MultiIndex)):
                # This branch indicates the passed as new_col
                # is actually a table-like data
                table = new_col
                from cudf._libxx.table import Table

                if isinstance(table, Table):
                    return self._parent._constructor_expanddim(
                        data=table._data
                    )
                else:
                    return self._parent._constructor_expanddim(
                        {index: value for index, value in enumerate(table)},
                        index=self._parent.index,
                    )
            elif isinstance(self._parent, Series):
                return Series(
                    new_col, index=self._parent.index, name=self._parent.name
                )
            elif isinstance(self._parent, Index):
                return as_index(new_col, name=self._parent.index)
            else:
                if self._parent is None:
                    return new_col
                else:
                    return self._parent._mimic_inplace(new_col, inplace=False)

    def __dir__(self):
        keys = dir(type(self))
        # TODO: Remove along with `__getattr__` above when all is ported
        return set(keys + dir(self._column.nvstrings))

    def len(self, **kwargs):
        """
        Computes the length of each element in the Series/Index.

        Returns
        -------
          Series or Index of int: A Series or Index of integer values
            indicating the length of each element in the Series or Index.
        """

        out_dev_arr = rmm.device_array(len(self._column), dtype="int32")
        ptr = libcudf.cudf.get_ctype_ptr(out_dev_arr)
        self._column.nvstrings.len(ptr)

        mask = None
        if self._column.has_nulls:
            mask = self._column.mask

        return self._return_or_inplace(
            column.build_column(
                Buffer(out_dev_arr), np.dtype("int32"), mask=mask
            ),
            **kwargs,
        )

    def cat(self, others=None, sep=None, na_rep=None, **kwargs):
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
        from cudf.core import DataFrame

        if sep is None:
            sep = ""

        from cudf._libxx.scalar import Scalar

        if others is None:
            data = cpp_join(self._column, Scalar(sep), Scalar(na_rep, "str"))
        else:
            others = _get_cols_list(others)
            if isinstance(others, list):
                cols = [self._column] + others
            else:
                cols = [self._column, others]
            data = cpp_concatenate(
                DataFrame({index: value for index, value in enumerate(cols)}),
                Scalar(sep),
                Scalar(na_rep, "str"),
            )

        out = self._return_or_inplace(data, **kwargs)
        if len(out) == 1 and others is None:
            out = out[0]
        return out

    def join(self, sep, na_rep="", **kwargs):
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.

        Parameters
        ----------
            sep : str
                Delimiter to use between list entries.

            na_rep : str
                This character will take the place of any null strings
                (not empty strings) in either list.

        Returns
        -------
        Series/Index of str dtype
            The list entries concatenated by intervening
            occurrences of the delimiter.

        """
        from cudf._libxx.scalar import Scalar
        from cudf.core.series import Series

        return Series(
            cpp_join(self._column, Scalar(sep), Scalar(na_rep)), **kwargs,
        )

    def extract(self, pat, flags=0, expand=True, **kwargs):
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

        out = self._column.nvstrings.extract(pat)
        if len(out) == 1 and expand is False:
            return self._return_or_inplace(out[0], **kwargs)
        else:
            kwargs.setdefault("expand", expand)
            return self._return_or_inplace(out, **kwargs)

    def contains(
        self, pat, case=True, flags=0, na=np.nan, regex=True, **kwargs
    ):
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

        out_dev_arr = rmm.device_array(len(self._column), dtype="bool")
        ptr = libcudf.cudf.get_ctype_ptr(out_dev_arr)
        self._column.nvstrings.contains(pat, regex=regex, devptr=ptr)

        mask = None
        if self._column.has_nulls:
            mask = self._column.mask

        return self._return_or_inplace(
            column.build_column(
                Buffer(out_dev_arr), dtype=np.dtype("bool"), mask=mask
            ),
            **kwargs,
        )

    def replace(
        self, pat, repl, n=-1, case=None, flags=0, regex=True, **kwargs
    ):
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

        return self._return_or_inplace(
            self._column.nvstrings.replace(pat, repl, n=n, regex=regex),
            **kwargs,
        )

    def lower(self, **kwargs):
        """
        Convert strings in the Series/Index to lowercase.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all strings converted to lowercase.
        """

        return self._return_or_inplace(
            self._column.nvstrings.lower(), **kwargs
        )

    # def slice(self, start=None, stop=None, step=None, **kwargs):
    #     """
    #     Returns a substring of each string.

    #     Parameters
    #     ----------
    #     start : int
    #         Beginning position of the string to extract.
    #         Default is beginning of the each string.
    #     stop : int
    #         Ending position of the string to extract.
    #         Default is end of each string.
    #     step : int
    #         Characters that are to be captured within the specified section.
    #         Default is every character.

    #     Returns
    #     -------
    #     Series/Index of str dtype
    #         A substring of each string.

    #     """

    #     return self._return_or_inplace(
    #         cpp_slice_strings(self._column, start, stop, step), **kwargs,
    #     )

    def isdecimal(self, **kwargs):
        """
        Returns a Series/Column/Index of boolean values with True for strings
        that contain only decimal characters -- those that can be used
        to extract base10 numbers.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_decimal(self._column))

    def isalnum(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only alpha-numeric characters.
        Equivalent to: isalpha() or isdigit() or isnumeric() or isdecimal()

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_alnum(self._column))

    def isalpha(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only alphabetic characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_alpha(self._column))

    def isdigit(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only decimal and digit characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_digit(self._column))

    def isnumeric(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only numeric characters. These include digit and
        numeric characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_numeric(self._column))

    def slice_from(self, starts=0, stops=0, **kwargs):
        """
        Return substring of each string using positions for each string.

        The starts and stops parameters are of Column type.

        Parameters
        ----------
        starts : Column
            Beginning position of each the string to extract.
            Default is beginning of the each string.
        stops : Column
            Ending position of the each string to extract.
            Default is end of each string.
            Use -1 to specify to the end of that string.

        Returns
        -------
        Series/Index of str dtype
            A substring of each string using positions for each string.

        """

        return self._return_or_inplace(
            cpp_slice_from(self._column, starts, stops), **kwargs
        )

    def slice_replace(self, start=None, stop=None, repl=None, **kwargs):
        """
        Replace the specified section of each string with a new string.

        Parameters
        ----------
        start : int
            Beginning position of the string to replace.
            Default is beginning of the each string.
        stop : int
            Ending position of the string to replace.
            Default is end of each string.
        repl : str
            String to insert into the specified position values.

        Returns
        -------
        Series/Index of str dtype
            A new string with the specified section of the string
            replaced with `repl` string.

        """
        if start is None:
            start = 0

        if stop is None:
            stop = -1

        if repl is None:
            repl = ""

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_slice_replace(self._column, start, stop, Scalar(repl)),
            **kwargs,
        )

    def insert(self, start=0, repl=None, **kwargs):
        """
        Insert the specified string into each string in the specified
        position.

        Parameters
        ----------
        start : int
            Beginning position of the string to replace.
            Default is beginning of the each string.
            Specify -1 to insert at the end of each string.
        repl : str
            String to insert into the specified position valus.

        Returns
        -------
        Series/Index of str dtype
            A new string series with the specified string
            inserted at the specified position.

        """
        if repl is None:
            repl = ""

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_string_insert(self._column, start, Scalar(repl)), **kwargs
        )

    # def get(self, i=0, **kwargs):
    #     """
    #     Returns the character specified in each string as a new string.
    #     The nvstrings returned contains a list of single character strings.

    #     Parameters
    #     ----------
    #     i : int
    #         The character position identifying the character
    #         in each string to return.

    #     Returns
    #     -------
    #     Series/Index of str dtype
    #         A new string series with character at the position
    #         `i` of each `i` inserted at the specified position.

    #     """

    #     return self._return_or_inplace(
    #         cpp_string_get(self._column, i), **kwargs
    #     )

    def split(self, pat=None, n=-1, expand=True, **kwargs):
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

        kwargs.setdefault("expand", expand)
        if pat is None:
            pat = " "

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_split(self._column, Scalar(pat), n), **kwargs,
        )

    def rsplit(self, pat=None, n=-1, expand=True, **kwargs):
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the end, at the
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
            raise NotImplementedError("`expand=False` is not yet supported")

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        kwargs.setdefault("expand", expand)
        if pat is None:
            pat = " "

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_rsplit(self._column, Scalar(pat), n), **kwargs,
        )

    def partition(self, sep=" ", expand=True, **kwargs):
        """
        Split the string at the first occurrence of sep.

        This method splits the string at the first occurrence
        of sep, and returns 3 elements containing the part
        before the separator, the separator itself, and the
        part after the separator. If the separator is not found,
        return 3 elements containing the string itself, followed
        by two empty strings.

        Parameters
        ----------
        sep : str, default ' ' (whitespace)
            String to split on.

        Returns
        -------
        DataFrame
            Returns a DataFrame

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        NotImplementedError if anything other than the default value is set.
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        kwargs.setdefault("expand", expand)
        if sep is None:
            sep = " "

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_partition(self._column, Scalar(sep)), **kwargs,
        )

    def rpartition(self, sep=" ", expand=True, **kwargs):
        """
        Split the string at the last occurrence of sep.

        This method splits the string at the last occurrence
        of sep, and returns 3 elements containing the part
        before the separator, the separator itself, and the
        part after the separator. If the separator is not
        found, return 3 elements containing two empty strings,
        followed by the string itself.

        Parameters
        ----------
        sep : str, default ' ' (whitespace)
            String to split on.

        Returns
        -------
        DataFrame
            Returns a DataFrame

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        NotImplementedError if anything other than the default value is set.
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        kwargs.setdefault("expand", expand)
        if sep is None:
            sep = " "

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_rpartition(self._column, Scalar(sep)), **kwargs,
        )

    def pad(self, width, side="left", fillchar=" ", **kwargs):
        """
        Pad strings in the Series/Index up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled with
            character defined in fillchar.

        side : {‘left’, ‘right’, ‘both’}, default ‘left’
            Side from which to fill resulting string.

        fillchar : str,  default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index with minimum number
            of char in object.

        """
        if not isinstance(fillchar, str):
            msg = (
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        try:
            side = PadSide[side.upper()]
        except KeyError:
            raise ValueError(
                "side has to be either one of {‘left’, ‘right’, ‘both’}"
            )

        return self._return_or_inplace(
            cpp_pad(self._column, width, fillchar, side), **kwargs
        )

    def zfill(self, width, **kwargs):
        """
        Pad strings in the Series/Index by prepending ‘0’ characters.

        Parameters
        ----------
        width : int
            Minimum length of resulting string;
            strings with length less than width
            be prepended with ‘0’ characters.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index with prepended ‘0’ characters.

        """
        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(
            cpp_zfill(self._column, width), **kwargs
        )

    def center(self, width, fillchar=" ", **kwargs):
        """
        Filling left and right side of strings in the Series/Index with an
        additional character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with fillchar.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if not isinstance(fillchar, str):
            msg = (
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(
            cpp_center(self._column, width, fillchar), **kwargs
        )

    def ljust(self, width, fillchar=" ", **kwargs):
        """
        Filling right side of strings in the Series/Index with an additional
        character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with fillchar.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if not isinstance(fillchar, str):
            msg = (
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(
            cpp_ljust(self._column, width, fillchar), **kwargs
        )

    def rjust(self, width, fillchar=" ", **kwargs):
        """
        Filling left side of strings in the Series/Index with an additional
        character.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with fillchar.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if not isinstance(fillchar, str):
            msg = (
                f"fillchar must be a character, not {type(fillchar).__name__}"
            )
            raise TypeError(msg)

        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(
            cpp_rjust(self._column, width, fillchar), **kwargs
        )

    def strip(self, to_strip=None, **kwargs):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of
        specified characters from each string in the Series/Index
        from left and right sides.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to be removed.
            All combinations of this set of characters
            will be stripped. If None then whitespaces are removed.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if to_strip is None:
            to_strip = ""

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_strip(self._column, Scalar(to_strip)), **kwargs
        )

    def lstrip(self, to_strip=None, **kwargs):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from
        each string in the Series/Index from left side.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will
            be stripped. If None then whitespaces are removed.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if to_strip is None:
            to_strip = ""

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_lstrip(self._column, Scalar(to_strip)), **kwargs
        )

    def rstrip(self, to_strip=None, **kwargs):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from each
        string in the Series/Index from right side.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to
            be removed. All combinations of this
            set of characters will be stripped.
            If None then whitespaces are removed.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        """
        if to_strip is None:
            to_strip = ""

        from cudf._libxx.scalar import Scalar

        return self._return_or_inplace(
            cpp_rstrip(self._column, Scalar(to_strip)), **kwargs
        )


class StringColumn(column.ColumnBase):
    """Implements operations for Columns of String type
    """

    def __init__(self, mask=None, size=None, offset=0, children=()):
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
        dtype = np.dtype("object")

        if size is None:
            if len(children) == 0:
                size = 0
            elif children[0].size == 0:
                size = 0
            else:
                # one less because the last element of offsets is the number of
                # bytes in the data buffer
                size = children[0].size - 1
            size = size - offset

        super().__init__(
            None, size, dtype, mask=mask, offset=offset, children=children
        )

        # TODO: Remove these once NVStrings is fully deprecated / removed
        self._nvstrings = None
        self._nvcategory = None
        self._indices = None

    @property
    def base_size(self):
        if len(self.base_children) == 0:
            return 0
        else:
            return int(
                (self.base_children[0].size - 1)
                / self.base_children[0].dtype.itemsize
            )

    def set_base_data(self, value):
        if value is not None:
            raise RuntimeError(
                "StringColumns do not use data attribute of Column, use "
                "`set_base_children` instead"
            )
        else:
            super().set_base_data(value)

    def set_base_mask(self, value):
        super().set_base_mask(value)

        # TODO: Remove these once NVStrings is fully deprecated / removed
        self._indices = None
        self._nvcategory = None
        self._nvstrings = None

    def set_base_children(self, value):
        # TODO: Implement dtype validation of the children here somehow
        super().set_base_children(value)

        # TODO: Remove these once NVStrings is fully deprecated / removed
        self._indices = None
        self._nvcategory = None
        self._nvstrings = None

    @property
    def children(self):
        if self._children is None:
            if len(self.base_children) == 0:
                self._children = ()
            elif self.offset == 0 and self.base_children[0].size == (
                self.size + 1
            ):
                self._children = self.base_children
            else:
                # First get the base columns for chars and offsets
                chars_column = self.base_children[1]
                offsets_column = self.base_children[0]

                # Shift offsets column by the parent offset.
                offsets_column = column.build_column(
                    data=offsets_column.base_data,
                    dtype=offsets_column.dtype,
                    mask=offsets_column.base_mask,
                    size=self.size + 1,
                    offset=self.offset,
                )

                # Now run a subtraction binary op to shift all of the offsets
                # by the respective number of characters relative to the
                # parent offset
                chars_offset = offsets_column[0]
                offsets_column = offsets_column.binary_operator(
                    "sub", offsets_column.dtype.type(chars_offset)
                )

                # Shift the chars offset by the new first element of the
                # offsets column
                chars_size = offsets_column[self.size]
                chars_column = column.build_column(
                    data=chars_column.base_data,
                    dtype=chars_column.dtype,
                    mask=chars_column.base_mask,
                    size=chars_size,
                    offset=chars_offset,
                )

                self._children = (offsets_column, chars_column)
        return self._children

    def __contains__(self, item):
        return True in self.str().contains(f"^{item}$")

    def __reduce__(self):
        cpumem = self.to_arrow()
        return column.as_column, (cpumem, False, np.dtype("object"))

    def str(self, parent=None):
        return StringMethods(self, parent=parent)

    def __sizeof__(self):
        n = 0
        if len(self.base_children) == 2:
            n += (
                self.base_children[0].__sizeof__()
                + self.base_children[1].__sizeof__()
            )
        if self.base_mask is not None:
            n += self.base_mask.size
        return n

    def _memory_usage(self, deep=False):
        if deep:
            return self.__sizeof__()
        else:
            return self.str().size() * self.dtype.itemsize

    def __len__(self):
        return self.size

    # TODO: Remove this once NVStrings is fully deprecated / removed
    @property
    def nvstrings(self):
        if self._nvstrings is None:
            if self.nullable:
                mask_ptr = self.mask_ptr
            else:
                mask_ptr = None
            if self.size == 0:
                self._nvstrings = nvstrings.to_device([])
            else:
                self._nvstrings = nvstrings.from_offsets(
                    self.children[1].data_ptr,
                    self.children[0].data_ptr,
                    self.size,
                    mask_ptr,
                    ncount=self.null_count,
                    bdevmem=True,
                )
        return self._nvstrings

    # TODO: Remove these once NVStrings is fully deprecated / removed
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
        # TODO: Remove these once NVStrings is fully deprecated / removed
        self._nvstrings = None
        self._nvcategory = None
        self._indices = None

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

    @property
    def _nbytes(self):
        if self.size == 0:
            return 0
        else:
            return self.children[1].size

    def as_numerical_column(self, dtype, **kwargs):

        mem_dtype = np.dtype(dtype)
        str_dtype = mem_dtype
        out_dtype = mem_dtype

        if mem_dtype.type is np.datetime64:
            if "format" not in kwargs:
                if len(self) > 0:
                    # infer on host from the first not na element
                    fmt = pd.core.tools.datetimes._guess_datetime_format(
                        self[self.notna()][0]
                    )
                    kwargs.update(format=fmt)
        kwargs.update(dtype=out_dtype)

        return _str_to_numeric_typecast_functions[str_dtype](self, **kwargs)

    def as_datetime_column(self, dtype, **kwargs):
        return self.as_numerical_column(dtype, **kwargs)

    def as_string_column(self, dtype, **kwargs):
        return self

    def to_arrow(self):
        if len(self) == 0:
            sbuf = np.empty(0, dtype="int8")
            obuf = np.empty(0, dtype="int32")
            nbuf = None
        else:
            sbuf = self.children[1].data.to_host_array().view("int8")
            obuf = self.children[0].data.to_host_array().view("int32")
            nbuf = None
            if self.null_count > 0:
                nbuf = self.mask.to_host_array().view("int8")
                nbuf = pa.py_buffer(nbuf)

        sbuf = pa.py_buffer(sbuf)
        obuf = pa.py_buffer(obuf)

        if self.null_count == len(self):
            return pa.NullArray.from_buffers(
                pa.null(), len(self), [pa.py_buffer((b""))], self.null_count
            )
        else:
            return pa.StringArray.from_buffers(
                len(self), obuf, sbuf, nbuf, self.null_count
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

        return self.to_arrow().to_pandas().values

    def serialize(self):
        header = {"null_count": self.null_count}
        header["type-serialized"] = pickle.dumps(type(self))
        frames = []
        sub_headers = []

        for item in self.children:
            sheader, sframes = item.serialize()
            sub_headers.append(sheader)
            frames.extend(sframes)

        if self.null_count > 0:
            frames.append(self.mask)

        header["subheaders"] = sub_headers
        header["frame_count"] = len(frames)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        # Deserialize the mask, value, and offset frames
        buffers = [Buffer(each_frame) for each_frame in frames]

        if header["null_count"] > 0:
            nbuf = buffers[2]
        else:
            nbuf = None

        children = []
        for h, b in zip(header["subheaders"], buffers[:2]):
            column_type = pickle.loads(h["type-serialized"])
            children.append(column_type.deserialize(h, [b]))

        col = column.build_column(
            data=None, dtype="str", mask=nbuf, children=tuple(children)
        )
        return col

    def unordered_compare(self, cmpop, rhs):
        return _string_column_binop(self, rhs, op=cmpop)

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*
        """
        to_replace = column.as_column(to_replace, dtype=self.dtype)
        replacement = column.as_column(replacement, dtype=self.dtype)
        return libcudfxx.replace.replace(self, to_replace, replacement)

    def fillna(self, fill_value):
        if not is_scalar(fill_value):
            fill_value = column.as_column(fill_value, dtype=self.dtype)
        return libcudfxx.replace.replace_nulls(self, fill_value)

    def _find_first_and_last(self, value):
        found_indices = self.str().contains(f"^{value}$")
        found_indices = libcudfxx.unary.cast(found_indices, dtype=np.int32)
        first = column.as_column(found_indices).find_first_value(1)
        last = column.as_column(found_indices).find_last_value(1)
        return first, last

    def find_first_value(self, value, closest=False):
        return self._find_first_and_last(value)[0]

    def find_last_value(self, value, closest=False):
        return self._find_first_and_last(value)[1]

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
            return lhs.str().cat(others=rhs)
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(binop, type(self), type(rhs)))

    def sum(self, dtype=None):
        # Should we be raising here? Pandas can't handle the mix of strings and
        # None and throws, but we already have a test that looks to ignore
        # nulls and returns anyway.

        # if self.null_count > 0:
        #     raise ValueError("Cannot get sum of string column with nulls")

        if len(self) == 0:
            return ""
        return decode(self.children[1].data.to_host_array(), encoding="utf-8")

    @property
    def is_unique(self):
        return len(self.unique()) == len(self)

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Strings are not yet supported via `__cuda_array_interface__`"
        )

    def _mimic_inplace(self, other_col, inplace=False):
        out = super()._mimic_inplace(other_col, inplace=inplace)
        if inplace:
            # TODO: Remove these once NVStrings is fully deprecated / removed
            self._nvstrings = other_col._nvstrings
            self._nvcategory = other_col._nvcategory
            self._indices = other_col._indices

        return out


def _string_column_binop(lhs, rhs, op):
    nvtx_range_push("CUDF_BINARY_OP", "orange")
    # Allocate output
    masked = lhs.nullable or rhs.nullable
    out = column.column_empty_like(lhs, dtype="bool", masked=masked)
    # Call and fix null_count
    _ = libcudf.binops.apply_op(lhs=lhs, rhs=rhs, out=out, op=op)
    nvtx_range_pop()
    return out


def _get_cols_list(others):
    from cudf.core import Series, Index
    from cudf.core.column import as_column

    if (
        is_list_like(others)
        and len(others) > 0
        and (
            is_list_like(others[0])
            or isinstance(others[0], (Series, Index, pd.Series, pd.Index))
        )
    ):
        """
        If others is a list-like object (in our case lists & tuples)
        just another Series/Index, great go ahead with concatenation.
        """
        cols_list = [as_column(frame, dtype="str") for frame in others]
        return cols_list
    elif others is not None:
        return as_column(others, dtype="str")
    else:
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarrary "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )
