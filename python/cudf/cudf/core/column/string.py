# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import functools
import pickle
import warnings
from codecs import decode

import numpy as np
import pandas as pd
import pyarrow as pa

import nvstrings

import cudf._lib as libcudf
import cudf._lib.string_casting as str_cast
from cudf._lib.nvtext.generate_ngrams import (
    generate_ngrams as cpp_generate_ngrams,
)
from cudf._lib.nvtext.ngrams_tokenize import (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from cudf._lib.nvtext.normalize import normalize_spaces as cpp_normalize_spaces
from cudf._lib.nvtext.tokenize import (
    count_tokens as cpp_count_tokens,
    tokenize as cpp_tokenize,
)
from cudf._lib.nvtx import annotate
from cudf._lib.strings.attributes import (
    code_points as cpp_code_points,
    count_characters as cpp_count_characters,
)
from cudf._lib.strings.capitalize import (
    capitalize as cpp_capitalize,
    title as cpp_title,
)
from cudf._lib.strings.case import (
    swapcase as cpp_swapcase,
    to_lower as cpp_to_lower,
    to_upper as cpp_to_upper,
)
from cudf._lib.strings.char_types import (
    all_float as cpp_all_float,
    all_int as cpp_all_int,
    is_alnum as cpp_is_alnum,
    is_alpha as cpp_is_alpha,
    is_decimal as cpp_is_decimal,
    is_digit as cpp_is_digit,
    is_lower as cpp_is_lower,
    is_numeric as cpp_is_numeric,
    is_space as cpp_isspace,
    is_upper as cpp_is_upper,
)
from cudf._lib.strings.combine import (
    concatenate as cpp_concatenate,
    join as cpp_join,
)
from cudf._lib.strings.contains import (
    contains_re as cpp_contains_re,
    count_re as cpp_count_re,
    match_re as cpp_match_re,
)
from cudf._lib.strings.convert.convert_urls import (
    url_decode as cpp_url_decode,
    url_encode as cpp_url_encode,
)
from cudf._lib.strings.extract import extract as cpp_extract
from cudf._lib.strings.find import (
    contains as cpp_contains,
    endswith as cpp_endswith,
    find as cpp_find,
    rfind as cpp_rfind,
    startswith as cpp_startswith,
)
from cudf._lib.strings.findall import findall as cpp_findall
from cudf._lib.strings.padding import (
    PadSide,
    center as cpp_center,
    ljust as cpp_ljust,
    pad as cpp_pad,
    rjust as cpp_rjust,
    zfill as cpp_zfill,
)
from cudf._lib.strings.replace import (
    insert as cpp_string_insert,
    replace as cpp_replace,
    replace_multi as cpp_replace_multi,
    slice_replace as cpp_slice_replace,
)
from cudf._lib.strings.replace_re import (
    replace_multi_re as cpp_replace_multi_re,
    replace_re as cpp_replace_re,
    replace_with_backrefs as cpp_replace_with_backrefs,
)
from cudf._lib.strings.split.partition import (
    partition as cpp_partition,
    rpartition as cpp_rpartition,
)
from cudf._lib.strings.split.split import (
    rsplit as cpp_rsplit,
    split as cpp_split,
)
from cudf._lib.strings.strip import (
    lstrip as cpp_lstrip,
    rstrip as cpp_rstrip,
    strip as cpp_strip,
)
from cudf._lib.strings.substring import (
    get as cpp_string_get,
    slice_from as cpp_slice_from,
    slice_strings as cpp_slice_strings,
)
from cudf._lib.strings.translate import translate as cpp_translate
from cudf._lib.strings.wrap import wrap as cpp_wrap
from cudf.core.buffer import Buffer
from cudf.core.column import column, column_empty, datetime
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
                from cudf._lib.table import Table

                if isinstance(table, Table):
                    return self._parent._constructor_expanddim(
                        data=table._data, index=self._parent.index
                    )
                else:
                    return self._parent._constructor_expanddim(
                        {index: value for index, value in enumerate(table)},
                        index=self._parent.index,
                    )
            elif isinstance(self._parent, Series):
                retain_index = kwargs.get("retain_index", True)
                if retain_index:
                    return Series(
                        new_col,
                        name=self._parent.name,
                        index=self._parent.index,
                    )
                else:
                    return Series(new_col, name=self._parent.name)
            elif isinstance(self._parent, Index):
                return as_index(new_col, name=self._parent.name)
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

        return self._return_or_inplace(
            cpp_count_characters(self._column), **kwargs,
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

        from cudf._lib.scalar import Scalar

        if others is None:
            data = cpp_join(self._column, Scalar(sep), Scalar(na_rep, "str"))
        else:
            other_cols = _get_cols_list(others)
            all_cols = [self._column] + other_cols
            data = cpp_concatenate(
                DataFrame(
                    {index: value for index, value in enumerate(all_cols)}
                ),
                Scalar(sep),
                Scalar(na_rep, "str"),
            )

        if len(data) == 1 and data.null_count == 1:
            data = [""]
        out = self._return_or_inplace(data, **kwargs)
        if len(out) == 1 and others is None:
            out = out.iloc[0]
        return out

    def join(self, sep):
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.
        """
        raise NotImplementedError(
            "Columns of arrays / lists are not yet " "supported"
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

        out = cpp_extract(self._column, pat)
        if out._num_columns == 1 and expand is False:
            return self._return_or_inplace(out._columns[0], **kwargs)
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

        from cudf._lib.scalar import Scalar

        return self._return_or_inplace(
            cpp_contains_re(self._column, pat)
            if regex is True
            else cpp_contains(self._column, Scalar(pat, "str")),
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
        pat : str or list-like
            String(s) to be replaced as a character sequence or regular
            expression.
        repl : str or list-like
            String(s) to be used as replacement.
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
        from cudf.core import Series, Index

        if (
            is_list_like(pat)
            or isinstance(pat, (Series, Index, pd.Series, pd.Index))
        ) and (
            is_list_like(repl)
            or isinstance(repl, (Series, Index, pd.Series, pd.Index))
        ):
            warnings.warn(
                "`n` parameter is not supported when \
                `pat` and `repl` are list-like inputs"
            )

            return self._return_or_inplace(
                cpp_replace_multi_re(
                    self._column, pat, column.as_column(repl, dtype="str")
                )
                if regex
                else cpp_replace_multi(
                    self._column,
                    column.as_column(pat, dtype="str"),
                    column.as_column(repl, dtype="str"),
                ),
                **kwargs,
            )
        # Pandas treats 0 as all
        if n == 0:
            n = -1
        from cudf._lib.scalar import Scalar

        # Pandas forces non-regex replace when pat is a single-character
        return self._return_or_inplace(
            cpp_replace_re(self._column, pat, Scalar(repl, "str"), n)
            if regex is True and len(pat) > 1
            else cpp_replace(
                self._column, Scalar(pat, "str"), Scalar(repl, "str"), n
            ),
            **kwargs,
        )

    def replace_with_backrefs(self, pat, repl, **kwargs):
        """
        Use the `repl` back-ref template to create a new string
        with the extracted elements found using the `pat` expression.

        Parameters
        ----------
        pat : str
            Regex with groupings to identify extract sections.
            This should not be a compiled regex.
        repl : str
            String template containing back-reference indicators.

        Returns
        -------
        Series/Index of str dtype
        """
        return self._return_or_inplace(
            cpp_replace_with_backrefs(self._column, pat, repl), **kwargs
        )

    def slice(self, start=None, stop=None, step=None, **kwargs):
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int
            Start position for slice operation.
        stop : int
            Stop position for slice operation.
        step : int
            Step size for slice operation.

        Returns
        -------
        Series/Index of str dtype
            Series or Index from sliced substring from
            original string object.

        """

        return self._return_or_inplace(
            cpp_slice_strings(self._column, start, stop, step), **kwargs,
        )

    def isdecimal(self, **kwargs):
        """
        Returns a Series/Column/Index of boolean values with True for strings
        that contain only decimal characters -- those that can be used
        to extract base10 numbers.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_decimal(self._column), **kwargs)

    def isalnum(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only alpha-numeric characters.
        Equivalent to: isalpha() or isdigit() or isnumeric() or isdecimal()

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_alnum(self._column), **kwargs)

    def isalpha(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only alphabetic characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_alpha(self._column), **kwargs)

    def isdigit(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only decimal and digit characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_digit(self._column), **kwargs)

    def isnumeric(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only numeric characters. These include digit and
        numeric characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_numeric(self._column), **kwargs)

    def isupper(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only upper-case characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_upper(self._column), **kwargs)

    def islower(self, **kwargs):
        """
        Returns a Series/Index of boolean values with True for strings
        that contain only lower-case characters.

        Returns
        -------
        Series/Index of bool dtype

        """
        return self._return_or_inplace(cpp_is_lower(self._column), **kwargs)

    def lower(self, **kwargs):
        """
        Convert strings in the Series/Index to lowercase.

        Returns
        -------
        Series/Index of str dtype
            A copy of the object with all strings converted to lowercase.

        """
        return self._return_or_inplace(cpp_to_lower(self._column), **kwargs)

    def upper(self, **kwargs):
        """
        Convert each string to uppercase.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["Hello, friend","Goodbye, friend"])
        >>> print(s.str.upper())
        ['HELLO, FRIEND', 'GOODBYE, FRIEND']

        """
        return self._return_or_inplace(cpp_to_upper(self._column), **kwargs)

    def capitalize(self, **kwargs):
        """
        Capitalize first character of each string.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello, friend","goodbye, friend"])
        >>> print(s.str.capitalize())
        ['Hello, friend", "Goodbye, friend"]

        """
        return self._return_or_inplace(cpp_capitalize(self._column), **kwargs)

    def swapcase(self, **kwargs):
        """
        Change each lowercase character to uppercase and vice versa.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["Hello, Friend","Goodbye, Friend"])
        >>> print(s.str.swapcase())
        ['hELLO, fRIEND', 'gOODBYE, fRIEND']

        """
        return self._return_or_inplace(cpp_swapcase(self._column), **kwargs)

    def title(self, **kwargs):
        """
        Uppercase the first letter of each letter after a space
        and lowercase the rest.
        This only applies to ASCII characters at this time.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["Hello friend","goodnight moon"])
        >>> print(s.str.title())
        ['Hello Friend', 'Goodnight Moon']

        """
        return self._return_or_inplace(cpp_title(self._column), **kwargs)

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

        from cudf._lib.scalar import Scalar

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

        from cudf._lib.scalar import Scalar

        return self._return_or_inplace(
            cpp_string_insert(self._column, start, Scalar(repl)), **kwargs
        )

    def get(self, i=0, **kwargs):
        """
        Extract element from each component at specified position.

        Parameters
        ----------
        i : int
            Position of element to extract.

        Returns
        -------
        Series/Index of str dtype

        """

        return self._return_or_inplace(
            cpp_string_get(self._column, i), **kwargs
        )

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
            pat = ""

        from cudf._lib.scalar import Scalar

        result_table = cpp_split(self._column, Scalar(pat, "str"), n)
        if len(result_table._data) == 1:
            if result_table._data[0].null_count == len(self._column):
                result_table = []
            elif self._column.null_count == len(self._column):
                result_table = [self._column.copy()]

        return self._return_or_inplace(result_table, **kwargs,)

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
            pat = ""

        from cudf._lib.scalar import Scalar

        result_table = cpp_rsplit(self._column, Scalar(pat), n)
        if len(result_table._data) == 1:
            if result_table._data[0].null_count == len(self._parent):
                result_table = []
            elif self._parent.null_count == len(self._parent):
                result_table = [self._column.copy()]

        return self._return_or_inplace(result_table, **kwargs)

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

        from cudf._lib.scalar import Scalar

        return self._return_or_inplace(
            cpp_partition(self._column, Scalar(sep)), **kwargs
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

        from cudf._lib.scalar import Scalar

        return self._return_or_inplace(
            cpp_rpartition(self._column, Scalar(sep)), **kwargs
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

        from cudf._lib.scalar import Scalar

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

        from cudf._lib.scalar import Scalar

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

        from cudf._lib.scalar import Scalar

        return self._return_or_inplace(
            cpp_rstrip(self._column, Scalar(to_strip)), **kwargs
        )

    def wrap(self, width, **kwargs):
        """
        Wrap long strings in the Series/Index to be formatted in
        paragraphs with length less than a given width.

        Parameters
        ----------
        width : int
            Maximum line width.

        Returns
        -------
        Series or Index

        Notes
        -----
        The parameters `expand_tabsbool`, `replace_whitespace`,
        `drop_whitespace`, `break_long_words`, `break_on_hyphens`,
        `expand_tabsbool` are not yet supported and will raise a
        NotImplementedError if they are set to any value.

        This method currently achieves behavior matching R’s
        stringr library str_wrap function, the equivalent
        pandas implementation can be obtained using the
        following parameter setting:

            expand_tabs = False

            replace_whitespace = True

            drop_whitespace = True

            break_long_words = False

            break_on_hyphens = False
        """
        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        expand_tabs = kwargs.get("expand_tabs", None)
        if expand_tabs is True:
            raise NotImplementedError("`expand_tabs=True` is not supported")
        elif expand_tabs is None:
            warnings.warn(
                "wrap current implementation defaults to `expand_tabs`=False"
            )

        replace_whitespace = kwargs.get("replace_whitespace", True)
        if not replace_whitespace:
            raise NotImplementedError(
                "`replace_whitespace=False` is not supported"
            )

        drop_whitespace = kwargs.get("drop_whitespace", True)
        if not drop_whitespace:
            raise NotImplementedError(
                "`drop_whitespace=False` is not supported"
            )

        break_long_words = kwargs.get("break_long_words", None)
        if break_long_words is True:
            raise NotImplementedError(
                "`break_long_words=True` is not supported"
            )
        elif break_long_words is None:
            warnings.warn(
                "wrap current implementation defaults to \
                    `break_long_words`=False"
            )

        break_on_hyphens = kwargs.get("break_on_hyphens", None)
        if break_long_words is True:
            raise NotImplementedError(
                "`break_on_hyphens=True` is not supported"
            )
        elif break_on_hyphens is None:
            warnings.warn(
                "wrap current implementation defaults to \
                    `break_on_hyphens`=False"
            )

        return self._return_or_inplace(cpp_wrap(self._column, width), **kwargs)

    def count(self, pat, flags=0, **kwargs):
        """
        Count occurrences of pattern in each string of the Series/Index.

        This function is used to count the number of times a particular
        regex pattern is repeated in each of the string elements of the Series.

        Parameters
        ----------
        pat : str
            Valid regular expression.

        Returns
        -------
        Series or Index

        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        return self._return_or_inplace(
            cpp_count_re(self._column, pat), **kwargs
        )

    def findall(self, pat, flags=0, **kwargs):
        """
        Find all occurrences of pattern or regular expression in the
        Series/Index.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.

        Returns
        -------
        DataFrame
            All non-overlapping matches of pattern or
            regular expression in each string of this Series/Index.

        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        kwargs.setdefault("expand", True)
        return self._return_or_inplace(
            cpp_findall(self._column, pat), **kwargs
        )

    def isempty(self, **kwargs):
        """
        Check whether each string is a an empty string.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.
        """
        return self._return_or_inplace(
            (self._parent == "").fillna(False), **kwargs
        )

    def isspace(self, **kwargs):
        """
        Check whether all characters in each string are whitespace.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.
        """
        return self._return_or_inplace(cpp_isspace(self._column), **kwargs)

    def endswith(self, pat, **kwargs):
        """
        Test if the end of each string element matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the end of each string element.

        """
        if "na" in kwargs:
            warnings.warn(
                "`na` parameter is not yet supported, \
                as cudf uses native strings instead of Python objects"
            )

        if pat is None:
            result_col = column.column_empty(
                len(self._column), dtype="bool", masked=True
            )
        else:
            from cudf._lib.scalar import Scalar

            result_col = cpp_endswith(self._column, Scalar(pat, "str"))

        return self._return_or_inplace(result_col, **kwargs)

    def startswith(self, pat, **kwargs):
        """
        Test if the start of each string element matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the start of each string element.

        """
        if "na" in kwargs:
            warnings.warn(
                "`na` parameter is not yet supported, \
                as cudf uses native strings instead of Python objects"
            )

        if pat is None:
            result_col = column.column_empty(
                len(self._column), dtype="bool", masked=True
            )
        else:
            from cudf._lib.scalar import Scalar

            result_col = cpp_startswith(self._column, Scalar(pat, "str"))

        return self._return_or_inplace(result_col, **kwargs)

    def find(self, sub, start=0, end=None, **kwargs):
        """
        Return lowest indexes in each strings in the Series/Index
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of int

        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        from cudf._lib.scalar import Scalar

        if end is None:
            end = -1

        result_col = cpp_find(self._column, Scalar(sub, "str"), start, end)

        return self._return_or_inplace(result_col, **kwargs)

    def rfind(self, sub, start=0, end=None, **kwargs):
        """
        Return highest indexes in each strings in the Series/Index
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of int

        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        from cudf._lib.scalar import Scalar

        if end is None:
            end = -1

        result_col = cpp_rfind(self._column, Scalar(sub, "str"), start, end)

        return self._return_or_inplace(result_col, **kwargs)

    def index(self, sub, start=0, end=None, **kwargs):
        """
        Return lowest indexes in each strings where the substring
        is fully contained between [start:end]. This is the same
        as str.find except instead of returning -1, it raises a ValueError
        when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of object

        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        from cudf._lib.scalar import Scalar

        if end is None:
            end = -1

        result_col = cpp_find(self._column, Scalar(sub, "str"), start, end)

        result = self._return_or_inplace(result_col, **kwargs)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result

    def rindex(self, sub, start=0, end=None, **kwargs):
        """
        Return highest indexes in each strings where the substring
        is fully contained between [start:end]. This is the same
        as str.rfind except instead of returning -1, it raises a ValueError
        when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched.

        start : int
            Left edge index.

        end : int
            Right edge index.

        Returns
        -------
        Series or Index of object

        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        from cudf._lib.scalar import Scalar

        if end is None:
            end = -1

        result_col = cpp_rfind(self._column, Scalar(sub, "str"), start, end)

        result = self._return_or_inplace(result_col, **kwargs)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result

    def match(self, pat, case=True, flags=0, **kwargs):
        """
        Determine if each string matches a regular expression.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.

        Returns
        -------
        Series or Index of boolean values.

        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        elif flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        if "na" in kwargs:
            warnings.warn(
                "`na` parameter is not yet supported, \
                as cudf uses native strings instead of Python objects"
            )

        return self._return_or_inplace(
            cpp_match_re(self._column, pat), **kwargs
        )

    def url_decode(self, **kwargs):
        """
        Returns a URL-decoded format of each string.
        No format checking is performed. All characters
        are expected to be encoded as UTF-8 hex values.

        Returns
        -------
        Series or Index.

        """

        return self._return_or_inplace(cpp_url_decode(self._column), **kwargs)

    def url_encode(self, **kwargs):
        """
        Returns a URL-encoded format of each string.
        No format checking is performed.
        All characters are encoded except for ASCII letters,
        digits, and these characters: ‘.’,’_’,’-‘,’~’.
        Encoding converts to hex using UTF-8 encoded bytes.

        Returns
        -------
        Series or Index.

        """
        return self._return_or_inplace(cpp_url_encode(self._column), **kwargs)

    def code_points(self, **kwargs):
        """
        Returns an array by filling it with the UTF-8 code point
        values for each character of each string.
        This function uses the len() method to determine
        the size of each sub-array of integers.

        Returns
        -------
        Series or Index.
        """
        from cudf.core.series import Series, Index

        new_col = cpp_code_points(self._column)
        if self._parent is None:
            return new_col
        elif isinstance(self._parent, Series):
            return Series(new_col, name=self._parent.name)
        elif isinstance(self._parent, Index):
            return column.as_index(new_col, name=self._parent.name)

    def translate(self, table, **kwargs):
        """
        Map all characters in the string through the given
        mapping table.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode
            ordinals, strings, or None.
            Unmapped characters are left untouched.
            str.maketrans() is a helper function for making translation tables.

        Returns
        -------
        Series or Index.
        """
        table = str.maketrans(table)
        return self._return_or_inplace(
            cpp_translate(self._column, table), **kwargs
        )

    def normalize_spaces(self, **kwargs):
        return self._return_or_inplace(
            cpp_normalize_spaces(self._column), **kwargs
        )

    def tokenize(self, delimiter="", **kwargs):
        delimiter = _massage_string_arg(delimiter, "delimiter", allow_col=True)
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_tokenize(self._column, delimiter), **kwargs
        )

    def token_count(self, delimiter="", **kwargs):
        delimiter = _massage_string_arg(delimiter, "delimiter", allow_col=True)
        return self._return_or_inplace(
            cpp_count_tokens(self._column, delimiter), **kwargs
        )

    def ngrams(self, n=2, separator="_", **kwargs):
        separator = _massage_string_arg(separator, "separator")
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_generate_ngrams(self._column, n, separator), **kwargs
        )

    def ngrams_tokenize(self, n=2, delimiter="", separator="_", **kwargs):
        delimiter = _massage_string_arg(delimiter, "delimiter")
        separator = _massage_string_arg(separator, "separator")
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_ngrams_tokenize(self._column, n, delimiter, separator),
            **kwargs,
        )


def _massage_string_arg(value, name, allow_col=False):
    from cudf._lib.scalar import Scalar
    from cudf._lib.column import Column
    from cudf.utils.dtypes import is_string_dtype

    if isinstance(value, str):
        return Scalar(value, dtype="str")

    if isinstance(value, Scalar) and is_string_dtype(value.dtype):
        return value

    allowed_types = ["Scalar"]

    if allow_col:
        if isinstance(value, list):
            return column.as_column(value, dtype="str")

        if isinstance(value, Column) and is_string_dtype(value.dtype):
            return value

        allowed_types.append("Column")

    raise ValueError(
        "Expected {} for {} but got {}".format(
            _expected_types_format(allowed_types), name, type(value)
        )
    )


def _expected_types_format(types):
    if len(types) == 1:
        return types[0]

    return ", ".join(types[:-1]) + ", or " + types[-1]


class StringColumn(column.ColumnBase):
    """Implements operations for Columns of String type
    """

    def __init__(
        self, mask=None, size=None, offset=0, null_count=None, children=()
    ):
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
            for child in children:
                assert child.offset == 0

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
            None,
            size,
            dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
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

    # TODO: Remove these once NVStrings is fully deprecated / removed
    @nvcategory.setter
    def nvcategory(self, nvc):
        self._nvcategory = nvc

    def _set_mask(self, value):
        # TODO: Remove these once NVStrings is fully deprecated / removed
        self._nvstrings = None
        self._nvcategory = None
        self._indices = None

        super()._set_mask(value)

    # TODO: Remove these once NVStrings is fully deprecated / removed
    @property
    def indices(self):
        if self._indices is None:
            out_col = column_empty(self.nvcategory.size(), dtype="int32")
            ptr = out_col.data_ptr
            self.nvcategory.values(devptr=ptr)
            self._indices = out_col.data_array_view
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
                    fmt = datetime.infer_format(self[self.notna()][0])
                    kwargs.update(format=fmt)
        kwargs.update(dtype=out_dtype)

        if str_dtype.kind in ("i"):
            if not cpp_all_int(self):
                raise ValueError("Cannot typecast to int")
        if str_dtype.kind in ("f"):
            if not cpp_all_float(self):
                raise ValueError("Cannot typecast to float")

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

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not allowed, \
            Conversion to GPU array in strings is not yet supported.\nTo \
            explicitly construct a host array, consider using .to_array()"
        )

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

    def find_and_replace(self, to_replace, replacement, all_nan):
        """
        Return col with *to_replace* replaced with *value*
        """
        to_replace = column.as_column(to_replace, dtype=self.dtype)
        replacement = column.as_column(replacement, dtype=self.dtype)
        return libcudf.replace.replace(self, to_replace, replacement)

    def fillna(self, fill_value):
        if not is_scalar(fill_value):
            fill_value = column.as_column(fill_value, dtype=self.dtype)
        return libcudf.replace.replace_nulls(self, fill_value, dtype="object")

    def _find_first_and_last(self, value):
        found_indices = self.str().contains(f"^{value}$")
        found_indices = libcudf.unary.cast(found_indices, dtype=np.int32)
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

    def binary_operator(self, op, rhs, reflect=False):
        lhs = self
        if reflect:
            lhs, rhs = rhs, lhs
        if isinstance(rhs, StringColumn) and op == "add":
            return lhs.str().cat(others=rhs)
        elif op in ("eq", "ne", "gt", "lt", "ge", "le"):
            return _string_column_binop(self, rhs, op=op, out_dtype="bool")
        else:
            msg = "{!r} operator not supported between {} and {}"
            raise TypeError(msg.format(op, type(self), type(rhs)))

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


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def _string_column_binop(lhs, rhs, op, out_dtype):
    out = libcudf.binaryop.binaryop(lhs=lhs, rhs=rhs, op=op, dtype=out_dtype)
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
        return [as_column(others, dtype="str")]
    else:
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarrary "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )
