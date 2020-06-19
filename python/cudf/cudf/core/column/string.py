# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import pickle
import warnings
from codecs import decode

import numpy as np
import pandas as pd
import pyarrow as pa

import cudf
import cudf._lib as libcudf
import cudf._lib.string_casting as str_cast
from cudf._lib.column import Column
from cudf._lib.nvtext.generate_ngrams import (
    generate_character_ngrams as cpp_generate_character_ngrams,
    generate_ngrams as cpp_generate_ngrams,
)
from cudf._lib.nvtext.ngrams_tokenize import (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from cudf._lib.nvtext.normalize import normalize_spaces as cpp_normalize_spaces
from cudf._lib.nvtext.replace import replace_tokens as cpp_replace_tokens
from cudf._lib.nvtext.tokenize import (
    character_tokenize as cpp_character_tokenize,
    count_tokens as cpp_count_tokens,
    tokenize as cpp_tokenize,
)
from cudf._lib.nvtx import annotate
from cudf._lib.scalar import Scalar, as_scalar
from cudf._lib.strings.attributes import (
    code_points as cpp_code_points,
    count_bytes as cpp_count_bytes,
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
    is_alnum as cpp_is_alnum,
    is_alpha as cpp_is_alpha,
    is_decimal as cpp_is_decimal,
    is_digit as cpp_is_digit,
    is_float as cpp_is_float,
    is_integer as cpp_is_integer,
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
    endswith_multiple as cpp_endswith_multiple,
    find as cpp_find,
    rfind as cpp_rfind,
    startswith as cpp_startswith,
    startswith_multiple as cpp_startswith_multiple,
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
from cudf.core.column import column, datetime
from cudf.utils import utils
from cudf.utils.dtypes import can_convert_to_column, is_scalar, is_string_dtype

_str_to_numeric_typecast_functions = {
    np.dtype("int8"): str_cast.stoi8,
    np.dtype("int16"): str_cast.stoi16,
    np.dtype("int32"): str_cast.stoi,
    np.dtype("int64"): str_cast.stol,
    np.dtype("uint8"): str_cast.stoui8,
    np.dtype("uint16"): str_cast.stoui16,
    np.dtype("uint32"): str_cast.stoui,
    np.dtype("uint64"): str_cast.stoul,
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
    np.dtype("uint8"): str_cast.ui8tos,
    np.dtype("uint16"): str_cast.ui16tos,
    np.dtype("uint32"): str_cast.uitos,
    np.dtype("uint64"): str_cast.ultos,
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
    def __init__(self, column, parent=None):
        """
        Vectorized string functions for Series and Index.

        This mimics pandas ``df.str`` interface. nulls stay null
        unless handled otherwise by a particular method.
        Patterned after Python’s string methods, with some
        inspiration from R’s stringr package.
        """
        self._column = column
        self._parent = parent

    def htoi(self):
        """
        Returns integer value represented by each hex string.
        String is interpretted to have hex (base-16) characters.

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1234", "ABCDEF", "1A2", "cafe"])
        >>> s.str.htoi()
        0        4660
        1    11259375
        2         418
        3       51966
        dtype: int64
        """

        out = str_cast.htoi(self._column)

        return self._return_or_inplace(out, inplace=False)

    def ip2int(self):
        """
        This converts ip strings to integers

        Returns
        -------
        Series/Index of str dtype

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["12.168.1.1", "10.0.0.1"])
        >>> s.str.ip2int()
        0    212336897
        1    167772161
        dtype: int64

        Returns 0's if any string is not an IP.

        >>> s = cudf.Series(["12.168.1.1", "10.0.0.1", "abc"])
        >>> s.str.ip2int()
        0    212336897
        1    167772161
        2            0
        dtype: int64
        """

        out = str_cast.ip2int(self._column)

        return self._return_or_inplace(out, inplace=False)

    def _return_or_inplace(self, new_col, **kwargs):
        """
        Returns an object of the type of the column owner or updates the column
        of the owner (Series or Index) to mimic an inplace operation
        """

        inplace = kwargs.get("inplace", False)

        if inplace:
            self._parent._mimic_inplace(new_col, inplace=True)
        else:
            expand = kwargs.get("expand", False)
            if expand or isinstance(
                self._parent, (cudf.DataFrame, cudf.MultiIndex)
            ):
                # This branch indicates the passed as new_col
                # is actually a table-like data
                table = new_col
                from cudf._lib.table import Table

                if isinstance(table, Table):
                    if isinstance(self._parent, cudf.Index):
                        idx = self._parent._constructor_expanddim._from_table(
                            table=table
                        )
                        idx.names = None
                        return idx
                    else:
                        return self._parent._constructor_expanddim(
                            data=table._data, index=self._parent.index
                        )
                else:
                    return self._parent._constructor_expanddim(
                        {index: value for index, value in enumerate(table)},
                        index=self._parent.index,
                    )
            elif isinstance(self._parent, cudf.Series):
                retain_index = kwargs.get("retain_index", True)
                if retain_index:
                    return cudf.Series(
                        new_col,
                        name=self._parent.name,
                        index=self._parent.index,
                    )
                else:
                    return cudf.Series(new_col, name=self._parent.name)
            elif isinstance(self._parent, cudf.Index):
                return cudf.core.index.as_index(
                    new_col, name=self._parent.name
                )
            else:
                if self._parent is None:
                    return new_col
                else:
                    return self._parent._mimic_inplace(new_col, inplace=False)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def len(self, **kwargs):
        """
        Computes the length of each element in the Series/Index.

        Returns : Series or Index of int
            A Series or Index of integer values
            indicating the length of each element in the Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["dog", "", "\\n", None])
        >>> s.str.len()
        0       3
        1       0
        2       1
        3    null
        dtype: int32
        """

        return self._return_or_inplace(
            cpp_count_characters(self._column), **kwargs,
        )

    def byte_count(self, **kwargs):
        """
        Computes the number of bytes of each string in the Series/Index.

        Returns : Series or Index of int
            A Series or Index of integer values
            indicating the number of bytes of each strings in the
            Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abc","d","ef"])
        >>> s.str.byte_count()
        0    3
        1    1
        2    2
        dtype: int32
        >>> s = cudf.Series(["Hello", "Bye", "Thanks 😊"])
        >>> s.str.byte_count()
        0     5
        1     3
        2    11
        dtype: int32
        """
        return self._return_or_inplace(
            cpp_count_bytes(self._column), **kwargs,
        )

    def cat(self, others=None, sep=None, na_rep=None, **kwargs):
        """
        Concatenate strings in the Series/Index with given separator.

        If ``others`` is specified, this function concatenates the Series/Index
        and elements of others element-wise. If others is not passed, then all
        values in the Series/Index are concatenated into a single string with
        a given sep.

        Parameters
        ----------
            others : Series or List of str
                Strings to be appended.
                The number of strings must match ``size()`` of this instance.
                This must be either a Series of string dtype or a Python
                list of strings.

            sep : str
                If specified, this separator will be appended to each string
                before appending the others.

            na_rep : str
                This character will take the place of any null strings
                (not empty strings) in either list.

                -  If ``na_rep`` is ``None``, and ``others`` is ``None``,
                   missing values in the Series/Index are
                   omitted from the result.

                -  If ``na_rep`` is ``None``, and ``others`` is
                   not ``None``, a row containing a missing value
                   in any of the columns (before concatenation)
                   will have a missing value in the result.

        Returns
        -------
        concat : str or Series/Index of str dtype
            If ``others`` is ``None``, ``str`` is returned,
            otherwise a ``Series/Index`` (same type as caller)
            of str dtype is returned.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'b', None, 'd'])
        >>> s.str.cat(sep=' ')
        'a b d'

        By default, NA values in the Series are ignored. Using na_rep, they
        can be given a representation:

        >>> s.str.cat(sep=' ', na_rep='?')
        'a b ? d'

        If others is specified, corresponding values are concatenated with
        the separator. Result will be a Series of strings.

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',')
        0     a,A
        1     b,B
        2    None
        3     d,D
        dtype: object

        Missing values will remain missing in the result, but can again be
        represented using na_rep

        >>> s.str.cat(['A', 'B', 'C', 'D'], sep=',', na_rep='-')
        0    a,A
        1    b,B
        2    -,C
        3    d,D
        dtype: object

        If sep is not specified, the values are concatenated without
        separation.

        >>> s.str.cat(['A', 'B', 'C', 'D'], na_rep='-')
        0    aA
        1    bB
        2    -C
        3    dD
        dtype: object
        """

        if sep is None:
            sep = ""

        if others is None:
            data = cpp_join(
                self._column, as_scalar(sep), as_scalar(na_rep, "str")
            )
        else:
            other_cols = _get_cols_list(others)
            all_cols = [self._column] + other_cols
            data = cpp_concatenate(
                cudf.DataFrame(
                    {index: value for index, value in enumerate(all_cols)}
                ),
                as_scalar(sep),
                as_scalar(na_rep, "str"),
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

        Raises : NotImplementedError
            Columns of arrays / lists are not yet supported.
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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a1', 'b2', 'c3'])
        >>> s.str.extract(r'([ab])(\d)')                                # noqa W605
              0     1
        0     a     1
        1     b     2
        2  None  None

        A pattern with one group will return a DataFrame with one
        column if expand=True.

        >>> s.str.extract(r'[ab](\d)', expand=True)                     # noqa W605
              0
        0     1
        1     2
        2  None

        A pattern with one group will return a Series if expand=False.

        >>> s.str.extract(r'[ab](\d)', expand=False)                    # noqa W605
        0       1
        1       2
        2    None
        dtype: object
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

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['Mouse', 'dog', 'house and parrot', '23', None])
        >>> s1
        0               Mouse
        1                 dog
        2    house and parrot
        3                  23
        4                None
        dtype: object
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4     null
        dtype: bool

        Returning an Index of booleans using only a literal pattern.

        >>> data = ['Mouse', 'dog', 'house and parrot', '23.0', np.NaN]
        >>> ind = cudf.core.index.StringIndex(data)
        >>> ind.str.contains('23', regex=False)
        Index(['False', 'False', 'False', 'True', 'null'], dtype='object')

        Returning ‘house’ or ‘dog’ when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4     null
        dtype: bool

        Returning any digit using regular expression.

        >>> s1.str.contains('\d', regex=True)                               # noqa W605
        0    False
        1    False
        2    False
        3     True
        4     null
        dtype: bool

        Ensure ``pat`` is a not a literal pattern when ``regex`` is set
        to True. Note in the following example one might expect
        only `s2[1]` and `s2[3]` to return True. However,
        ‘.0’ as a regex matches any character followed by a 0.

        >>> s2 = cudf.Series(['40', '40.0', '41', '41.0', '35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        dtype: bool
        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        elif flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")
        elif na is not np.nan:
            raise NotImplementedError("`na` parameter is not yet supported")

        return self._return_or_inplace(
            cpp_contains_re(self._column, pat)
            if regex is True
            else cpp_contains(self._column, as_scalar(pat, "str")),
            **kwargs,
        )

    def replace(
        self, pat, repl, n=-1, case=None, flags=0, regex=True, **kwargs
    ):
        """
        Replace occurrences of pattern/regex in the Series/Index with some
        other string. Equivalent to `str.replace()
        <https://docs.python.org/3/library/stdtypes.html#str.replace>`_
        or `re.sub()
        <https://docs.python.org/3/library/re.html#re.sub>`_.

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
        a `NotImplementedError` if anything other than the default value
        is set.

        Examples
        --------

        >>> import cudf
        >>> s = cudf.Series(['foo', 'fuz', None])
        >>> s
        0     foo
        1     fuz
        2    None
        dtype: object

        When pat is a string and regex is True (the default), the given pat
        is compiled as a regex. When repl is a string, it replaces matching
        regex patterns as with ``re.sub()``. NaN value(s) in the Series
        are left as is:

        >>> s.str.replace('f.', 'ba', regex=True)
        0     bao
        1     baz
        2    None
        dtype: object

        When pat is a string and `regex` is False, every pat is replaced
        with repl as with ``str.replace()``:

        >>> s.str.replace('f.', 'ba', regex=False)
        0     foo
        1     fuz
        2    None
        dtype: object
        """
        if case is not None:
            raise NotImplementedError("`case` parameter is not yet supported")
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        if can_convert_to_column(pat) and can_convert_to_column(repl):
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

        # Pandas forces non-regex replace when pat is a single-character
        return self._return_or_inplace(
            cpp_replace_re(self._column, pat, as_scalar(repl, "str"), n)
            if regex is True and len(pat) > 1
            else cpp_replace(
                self._column, as_scalar(pat, "str"), as_scalar(repl, "str"), n
            ),
            **kwargs,
        )

    def replace_with_backrefs(self, pat, repl, **kwargs):
        """
        Use the ``repl`` back-ref template to create a new string
        with the extracted elements found using the ``pat`` expression.

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["A543","Z756"])
        >>> s.str.replace_with_backrefs('(\\d)(\\d)', 'V\\2\\1')
        0    AV453
        1    ZV576
        dtype: object
        """
        return self._return_or_inplace(
            cpp_replace_with_backrefs(self._column, pat, repl), **kwargs
        )

    def slice(self, start=None, stop=None, step=None, **kwargs):
        """
        Slice substrings from each element in the Series or Index.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        Series/Index of str dtype
            Series or Index from sliced substring from
            original string object.

        See also
        --------
        slice_replace
            Replace a slice with a string.

        get
            Return element at position. Equivalent
            to ``Series.str.slice(start=i, stop=i+1)``
            with ``i`` being the position.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["koala", "fox", "chameleon"])
        >>> s
        0        koala
        1          fox
        2    chameleon
        dtype: object
        >>> s.str.slice(start=1)
        0        oala
        1          ox
        2    hameleon
        dtype: object
        >>> s.str.slice(start=-1)
        0    a
        1    x
        2    n
        dtype: object
        >>> s.str.slice(stop=2)
        0    ko
        1    fo
        2    ch
        dtype: object
        >>> s.str.slice(step=2)
        0      kaa
        1       fx
        2    caeen
        dtype: object
        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     f
        2    cm
        dtype: object
        """

        return self._return_or_inplace(
            cpp_slice_strings(self._column, start, stop, step), **kwargs,
        )

    def isinteger(self, **kwargs):
        """
        Check whether all characters in each string form integer.

        If a string has zero characters, False is returned for
        that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1", "0.1", "+100", "-15", "abc"])
        >>> s.str.isinteger()
        0     True
        1    False
        2     True
        3     True
        4    False
        dtype: bool
        >>> s = cudf.Series(["this is plan text", "", "10 10"])
        >>> s.str.isinteger()
        0    False
        1    False
        2    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_integer(self._column), **kwargs)

    def isfloat(self, **kwargs):
        """
        Check whether all characters in each string form floating value.

        If a string has zero characters, False is returned for
        that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1.1", "0.123213", "+0.123", "-100.0001", "234",
        ... "3-"])
        >>> s.str.isfloat()
        0     True
        1     True
        2     True
        3     True
        4     True
        5    False
        dtype: bool
        >>> s = cudf.Series(["this is plain text", "\t\n", "9.9", "9.9.9"])
        >>> s.str.isfloat()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_float(self._column), **kwargs)

    def isdecimal(self, **kwargs):
        """
        Check whether all characters in each string are decimal.

        This is equivalent to running the Python string method
        `str.isdecimal()
        <https://docs.python.org/3/library/stdtypes.html#str.isdecimal>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned for
        that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s3 = cudf.Series(['23', '³', '⅕', ''])

        The s3.str.isdecimal method checks for characters used to form
        numbers in base 10.

        >>> s3.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_decimal(self._column), **kwargs)

    def isalnum(self, **kwargs):
        """
        Check whether all characters in each string are alphanumeric.

        This is equivalent to running the Python string method
        `str.isalnum()
        <https://docs.python.org/3/library/stdtypes.html#str.isalnum>`_
        for each element of the Series/Index. If a string has zero
        characters, False is returned for that check.

        Equivalent to: ``isalpha() or isdigit() or isnumeric() or isdecimal()``

        Returns : Series or Index of bool
            Series or Index of boolean values with the
            same length as the original Series/Index.

        See also
        --------
        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalnum()
        0     True
        1     True
        2     True
        3    False
        dtype: bool

        Note that checks against characters mixed with
        any additional punctuation or whitespace will
        evaluate to false for an alphanumeric check.

        >>> s2 = cudf.Series(['A B', '1.5', '3,000'])
        >>> s2.str.isalnum()
        0    False
        1    False
        2    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_alnum(self._column), **kwargs)

    def isalpha(self, **kwargs):
        """
        Check whether all characters in each string are alphabetic.

        This is equivalent to running the Python string method
        `str.isalpha()
        <https://docs.python.org/3/library/stdtypes.html#str.isalpha>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same length
            as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isalpha()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_alpha(self._column), **kwargs)

    def isdigit(self, **kwargs):
        """
        Check whether all characters in each string are digits.

        This is equivalent to running the Python string method
        `str.isdigit()
        <https://docs.python.org/3/library/stdtypes.html#str.isdigit>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['23', '³', '⅕', ''])

        The ``s.str.isdigit`` method is the same as ``s.str.isdecimal`` but
        also includes special digits, like superscripted and
        subscripted digits in unicode.

        >>> s.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_digit(self._column), **kwargs)

    def isnumeric(self, **kwargs):
        """
        Check whether all characters in each string are numeric.

        This is equivalent to running the Python string method
        `str.isnumeric()
        <https://docs.python.org/3/library/stdtypes.html#str.isnumeric>`_
        for each element of the Series/Index. If a
        string has zero characters, False is returned for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s1 = cudf.Series(['one', 'one1', '1', ''])
        >>> s1.str.isnumeric()
        0    False
        1    False
        2     True
        3    False
        dtype: bool

        The ``s1.str.isnumeric`` method is the same as ``s2.str.isdigit`` but
        also includes other characters that can represent
        quantities such as unicode fractions.

        >>> s2 = pd.Series(['23', '³', '⅕', ''])
        >>> s2.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_numeric(self._column), **kwargs)

    def isupper(self, **kwargs):
        """
        Check whether all characters in each string are uppercase.

        This is equivalent to running the Python string method
        `str.isupper()
        <https://docs.python.org/3/library/stdtypes.html#str.isupper>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isspace
            Check whether all characters are whitespace.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.isupper()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_upper(self._column), **kwargs)

    def islower(self, **kwargs):
        """
        Check whether all characters in each string are lowercase.

        This is equivalent to running the Python string method
        `str.islower()
        <https://docs.python.org/3/library/stdtypes.html#str.islower>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        isspace
            Check whether all characters are whitespace.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.islower()
        0     True
        1    False
        2    False
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_lower(self._column), **kwargs)

    def lower(self, **kwargs):
        """
        Converts all characters to lowercase.

        Equivalent to `str.lower()
        <https://docs.python.org/3/library/stdtypes.html#str.lower>`_.

        Returns : Series or Index of object
            A copy of the object with all strings converted to lowercase.

        See also
        --------
        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and remaining
            to lowercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.lower()
        0                 lower
        1              capitals
        2    this is a sentence
        3              swapcase
        dtype: object
        """
        return self._return_or_inplace(cpp_to_lower(self._column), **kwargs)

    def upper(self, **kwargs):
        """
        Convert each string to uppercase.
        This only applies to ASCII characters at this time.

        Equivalent to `str.upper()
        <https://docs.python.org/3/library/stdtypes.html#str.upper>`_.

        Returns : Series or Index of object

        See also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and
            remaining to lowercase.

        capitalize
            Converts first character to uppercase and remaining to
            lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object
        >>> s.str.upper()
        0                 LOWER
        1              CAPITALS
        2    THIS IS A SENTENCE
        3              SWAPCASE
        dtype: object
        """
        return self._return_or_inplace(cpp_to_upper(self._column), **kwargs)

    def capitalize(self, **kwargs):
        """
        Convert strings in the Series/Index to be capitalized.
        This only applies to ASCII characters at this time.

        Returns
        -------
        Series or Index of object

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.capitalize()
        0                 Lower
        1              Capitals
        2    This is a sentence
        3              Swapcase
        dtype: object
        >>> s = cudf.Series(["hello, friend","goodbye, friend"])
        >>> s.str.capitalize()
        0      Hello, friend
        1    Goodbye, friend
        dtype: object
        """
        return self._return_or_inplace(cpp_capitalize(self._column), **kwargs)

    def swapcase(self, **kwargs):
        """
        Change each lowercase character to uppercase and vice versa.
        This only applies to ASCII characters at this time.

        Equivalent to `str.swapcase()
        <https://docs.python.org/3/library/stdtypes.html#str.swapcase>`_.

        Returns : Series or Index of object

        See also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        title
            Converts first character of each word to uppercase and remaining
            to lowercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object
        >>> s.str.swapcase()
        0                 LOWER
        1              capitals
        2    THIS IS A SENTENCE
        3              sWaPcAsE
        dtype: object
        """
        return self._return_or_inplace(cpp_swapcase(self._column), **kwargs)

    def title(self, **kwargs):
        """
        Uppercase the first letter of each letter after a space
        and lowercase the rest.
        This only applies to ASCII characters at this time.

        Equivalent to `str.title()
        <https://docs.python.org/3/library/stdtypes.html#str.title>`_.

        Returns : Series or Index of object

        See also
        --------
        lower
            Converts all characters to lowercase.

        upper
            Converts all characters to uppercase.

        capitalize
            Converts first character to uppercase and remaining to lowercase.

        swapcase
            Converts uppercase to lowercase and lowercase to uppercase.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s = cudf.Series(data)
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        dtype: object
        >>> s.str.title()
        0                 Lower
        1              Capitals
        2    This Is A Sentence
        3              Swapcase
        dtype: object
        """
        return self._return_or_inplace(cpp_title(self._column), **kwargs)

    def slice_from(self, starts, stops, **kwargs):
        """
        Return substring of each string using positions for each string.

        The starts and stops parameters are of Column type.

        Parameters
        ----------
        starts : Series
            Beginning position of each the string to extract.
            Default is beginning of the each string.
        stops : Series
            Ending position of the each string to extract.
            Default is end of each string.
            Use -1 to specify to the end of that string.

        Returns
        -------
        Series/Index of str dtype
            A substring of each string using positions for each string.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello","there"])
        >>> s
        0    hello
        1    there
        dtype: object
        >>> starts = cudf.Series([1, 3])
        >>> stops = cudf.Series([5, 5])
        >>> s.str.slice_from(starts, stops)
        0    ello
        1      re
        dtype: object
        """

        return self._return_or_inplace(
            cpp_slice_from(
                self._column, column.as_column(starts), column.as_column(stops)
            ),
            **kwargs,
        )

    def slice_replace(self, start=None, stop=None, repl=None, **kwargs):
        """
        Replace the specified section of each string with a new string.

        Parameters
        ----------
        start : int, optional
            Beginning position of the string to replace.
            Default is beginning of the each string.
        stop : int, optional
            Ending position of the string to replace.
            Default is end of each string.
        repl : str, optional
            String to insert into the specified position values.

        Returns
        -------
        Series/Index of str dtype
            A new string with the specified section of the string
            replaced with `repl` string.

        See also
        --------
        slice
            Just slicing without replacement.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        dtype: object

        Specify just `start`, meaning replace `start` until the `end` of
        the string with `repl`.

        >>> s.str.slice_replace(1, repl='X')
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        dtype: object

        Specify just `stop`, meaning the `start` of the string to `stop`
        is replaced with `repl`, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl='X')
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        dtype: object

        Specify `start` and `stop`, meaning the slice from `start`
        to `stop` is replaced with `repl`. Everything before or
        after `start` and `stop` is included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl='X')
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        dtype: object
        """
        if start is None:
            start = 0

        if stop is None:
            stop = -1

        if repl is None:
            repl = ""

        return self._return_or_inplace(
            cpp_slice_replace(self._column, start, stop, as_scalar(repl)),
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
            String to insert into the specified position value.

        Returns
        -------
        Series/Index of str dtype
            A new string series with the specified string
            inserted at the specified position.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abcdefghij", "0123456789"])
        >>> s.str.insert(2, '_')
        0    ab_cdefghij
        1    01_23456789
        dtype: object

        When no `repl` is passed, nothing is inserted.

        >>> s.str.insert(2)
        0    abcdefghij
        1    0123456789
        dtype: object

        Negative values are also supported for `start`.

        >>> s.str.insert(-1,'_')
        0    abcdefghij_
        1    0123456789_
        dtype: object
        """
        if repl is None:
            repl = ""

        return self._return_or_inplace(
            cpp_string_insert(self._column, start, as_scalar(repl)), **kwargs
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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids", "cudf"])
        >>> s
        0    hello world
        1         rapids
        2           cudf
        dtype: object
        >>> s.str.get(10)
        0    d
        1
        2
        dtype: object
        >>> s.str.get(1)
        0    e
        1    a
        2    u
        dtype: object

        ``get`` also accepts negative index number.

        >>> s.str.get(-1)
        0    d
        1    s
        2    f
        dtype: object
        """

        return self._return_or_inplace(
            cpp_string_get(self._column, i), **kwargs
        )

    def split(self, pat=None, n=-1, expand=None, **kwargs):
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the beginning, at the
        specified delimiter string. Equivalent to `str.split()
        <https://docs.python.org/3/library/stdtypes.html#str.split>`_.

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

        See also
        --------
        rsplit
            Splits string around given separator/delimiter, starting from
            the right.

        str.split
            Standard library version for split.

        str.rsplit
            Standard library version for rsplit.

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        NotImplementedError if anything other than the default value
        is set. The handling of the n keyword depends on the number
        of found splits:

            - If found splits > n, make first n splits only
            - If found splits <= n, make all splits
            - If for a certain row the number of found
              splits < n, append None for padding up to n

        Examples
        --------
        >>> import cudf
        >>> data = ["this is a regular sentence", "https://docs.python.org/index.html", None]       # noqa E501
        >>> s = cudf.Series(data)
        >>> s
        0            this is a regular sentence
        1    https://docs.python.org/index.html
        2                                  None
        dtype: object

        The `n` parameter can be used to limit the number of
        splits on the delimiter.

        >>> s.str.split(n=2)
                                            0     1                   2
        0                                this    is  a regular sentence
        1  https://docs.python.org/index.html  None                None
        2                                None  None                None

        The `pat` parameter can be used to split by other characters.

        >>> s.str.split(pat = "/")
                                    0     1                2           3
        0  this is a regular sentence  None             None        None
        1                      https:        docs.python.org  index.html
        2                        None  None             None        None
        """
        if expand is None:
            expand = True
            warnings.warn("`expand` parameter defatults to True.")
        elif expand is not True:
            raise NotImplementedError(
                "`expand=False` setting is not supported yet"
            )

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        kwargs.setdefault("expand", expand)
        if pat is None:
            pat = ""

        result_table = cpp_split(self._column, as_scalar(pat, "str"), n)
        if len(result_table._data) == 1:
            if result_table._data[0].null_count == len(self._column):
                result_table = []
            elif self._column.null_count == len(self._column):
                result_table = [self._column.copy()]

        return self._return_or_inplace(result_table, **kwargs,)

    def rsplit(self, pat=None, n=-1, expand=None, **kwargs):
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series/Index from the end, at the
        specified delimiter string. Equivalent to `str.rsplit()
        <https://docs.python.org/3/library/stdtypes.html#str.rsplit>`_.

        Parameters
        ----------
        pat : str, default ' ' (space)
            String to split on, does not yet support regular expressions.
        n : int, default -1 (all)
            Limit number of splits in output. `None`, 0, and -1 will all be
            interpreted as "all splits".

        Returns
        -------
        DataFrame or MultiIndex
            Returns a DataFrame/MultiIndex with each split as a column.

        See also
        --------
        split
            Split strings around given separator/delimiter.

        str.split
            Standard library version for split.

        str.rsplit
            Standard library version for rsplit.

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        `NotImplementedError` if anything other than the default value is
        set. The handling of the n keyword depends on the number of
        found splits:
            - If found splits > n, make first n splits only
            - If found splits <= n, make all splits
            - If for a certain row the number of found splits < n,
              append None for padding up to n.

        Examples
        --------
        >>> import cudf
        >>> data = ["this is a regular sentence","https://docs.python.org/3/tutorial/index.html",None]      # noqa E501
        >>> s = cudf.Series(data)
        >>> s.str.rsplit(n=2)
                                                       0        1         2
        0                                      this is a  regular  sentence
        1  https://docs.python.org/3/tutorial/index.html     None      None
        2                                           None     None      None

        For slightly more complex use cases like splitting the
        html document name from a url, a combination of parameter
        settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True)
                                            0           1
        0          this is a regular sentence        None
        1  https://docs.python.org/3/tutorial  index.html
        2                                None        None
        """
        if expand is None:
            expand = True
            warnings.warn("`expand` parameter defatults to True.")
        elif expand is not True:
            raise NotImplementedError(
                "`expand=False` setting is not supported yet"
            )

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        kwargs.setdefault("expand", expand)
        if pat is None:
            pat = ""

        result_table = cpp_rsplit(self._column, as_scalar(pat), n)
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
        DataFrame or MultiIndex
            Returns a DataFrame / MultiIndex

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        `NotImplementedError` if anything other than the default value is set.

        See also
        --------
        rpartition
            Split the string at the last occurrence of sep.

        split
            Split strings around given separators.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Linda van der Berg', 'George Pitt-Rivers'])
        >>> s
        0    Linda van der Berg
        1    George Pitt-Rivers
        dtype: object

        >>> s.str.partition()
                0  1             2
        0   Linda     van der Berg
        1  George      Pitt-Rivers

        To partition by something different than a space:

        >>> s.str.partition('-')
                            0  1       2
        0  Linda van der Berg
        1         George Pitt  -  Rivers

        Also available on indices:

        >>> idx = cudf.core.index.StringIndex(['X 123', 'Y 999'])
        >>> idx
        StringIndex(['X 123' 'Y 999'], dtype='object')

        Which will create a MultiIndex:

        >>> idx.str.partition()
        MultiIndex(levels=[0    X
        1    Y
        dtype: object, 0
        dtype: object, 0    123
        1    999
        dtype: object],
        codes=   0  1  2
        0  0  0  0
        1  1  0  1)
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        kwargs.setdefault("expand", expand)
        if sep is None:
            sep = " "

        return self._return_or_inplace(
            cpp_partition(self._column, as_scalar(sep)), **kwargs
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
        DataFrame or MultiIndex
            Returns a DataFrame / MultiIndex

        Notes
        -----
        The parameter `expand` is not yet supported and will raise a
        `NotImplementedError` if anything other than the default value is set.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Linda van der Berg', 'George Pitt-Rivers'])
        >>> s
        0    Linda van der Berg
        1    George Pitt-Rivers
        dtype: object
        >>> s.str.rpartition()
                    0  1            2
        0  Linda van der            Berg
        1         George     Pitt-Rivers

        Also available on indices:

        >>> idx = cudf.core.index.StringIndex(['X 123', 'Y 999'])
        >>> idx
        StringIndex(['X 123' 'Y 999'], dtype='object')

        Which will create a MultiIndex:

        >>> idx.str.rpartition()
        MultiIndex(levels=[0    X
        1    Y
        dtype: object, 0
        dtype: object, 0    123
        1    999
        dtype: object],
        codes=   0  1  2
        0  0  0  0
        1  1  0  1)
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        kwargs.setdefault("expand", expand)
        if sep is None:
            sep = " "

        return self._return_or_inplace(
            cpp_rpartition(self._column, as_scalar(sep)), **kwargs
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
        Series/Index of object
            Returns Series or Index with minimum number
            of char in object.

        See also
        --------
        rjust
            Fills the left side of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='left')``.

        ljust
            Fills the right side of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='right')``.

        center
            Fills boths sides of strings with an arbitrary character.
            Equivalent to ``Series.str.pad(side='both')``.

        zfill
            Pad strings in the Series/Index by prepending ‘0’ character.
            Equivalent to ``Series.str.pad(side='left', fillchar='0')``.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["caribou", "tiger"])

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        dtype: object

        >>> s.str.pad(width=10, side='right', fillchar='-')
        0    caribou---
        1    tiger-----
        dtype: object

        >>> s.str.pad(width=10, side='both', fillchar='-')
        0    -caribou--
        1    --tiger---
        dtype: object
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

        Strings in the Series/Index are padded with ‘0’ characters
        on the left of the string to reach a total string length
        width. Strings in the Series/Index with length greater
        or equal to width are unchanged.

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

        See also
        --------
        rjust
            Fills the left side of strings with an arbitrary character.

        ljust
            Fills the right side of strings with an arbitrary character.

        pad
            Fills the specified sides of strings with an arbitrary character.

        center
            Fills boths sides of strings with an arbitrary character.

        Notes
        -----
        Differs from `str.zfill()
        <https://docs.python.org/3/library/stdtypes.html#str.zfill>`_
        which has special handling for ‘+’/’-‘ in the string.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['-1', '1', '1000',  None])
        >>> s
        0      -1
        1       1
        2    1000
        3    None
        dtype: object

        Note that ``None`` is not string, therefore it is converted
        to ``None``. The minus sign in ``'-1'`` is treated as a
        regular character and the zero is added to the left
        of it (`str.zfill()
        <https://docs.python.org/3/library/stdtypes.html#str.zfill>`_
        would have moved it to the left). ``1000`` remains unchanged as
        it is longer than width.

        >>> s.str.zfill(3)
        0     0-1
        1     001
        2    1000
        3    None
        dtype: object
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

        fillchar : str, default is ' ' (whitespace)
            Additional character for filling.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'b', None, 'd'])
        >>> s.str.center(1)
        0       a
        1       b
        2    None
        3       d
        dtype: object
        >>> s.str.center(1, fillchar='-')
        0       a
        1       b
        2    None
        3       d
        dtype: object
        >>> s.str.center(2, fillchar='-')
        0      a-
        1      b-
        2    None
        3      d-
        dtype: object
        >>> s.str.center(5, fillchar='-')
        0    --a--
        1    --b--
        2     None
        3    --d--
        dtype: object
        >>> s.str.center(6, fillchar='-')
        0    --a---
        1    --b---
        2      None
        3    --d---
        dtype: object
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
        character. Equivalent to `str.ljust()
        <https://docs.python.org/3/library/stdtypes.html#str.ljust>`_.

        Parameters
        ----------
        width : int
            Minimum width of resulting string;
            additional characters will be filled
            with ``fillchar``.

        fillchar : str, default ' ' (whitespace)
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series/Index of str dtype
            Returns Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids ai"])
        >>> s.str.ljust(10, fillchar="_")
        0    hello world
        1     rapids ai_
        dtype: object
        >>> s = cudf.Series(["a", "",  "ab", "__"])
        >>> s.str.ljust(1, fillchar="-")
        0     a
        1     -
        2    ab
        3    __
        dtype: object
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
        character. Equivalent to `str.rjust()
        <https://docs.python.org/3/library/stdtypes.html#str.rjust>`_.

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["hello world", "rapids ai"])
        >>> s.str.rjust(20, fillchar="_")
        0    _________hello world
        1    ___________rapids ai
        dtype: object
        >>> s = cudf.Series(["a", "",  "ab", "__"])
        >>> s.str.rjust(1, fillchar="-")
        0     a
        1     -
        2    ab
        3    __
        dtype: object
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
        from left and right sides. Equivalent to `str.strip()
        <https://docs.python.org/3/library/stdtypes.html#str.strip>`_.

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

        See also
        --------
        lstrip
            Remove leading characters in Series/Index.

        rstrip
            Remove trailing characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\\n', '3. Cat?\\t', None])
        >>> s
        0    1. Ant.
        1    2. Bee!\\n
        2    3. Cat?\\t
        3         None
        dtype: object
        >>> s.str.strip()
        0    1. Ant.
        1    2. Bee!
        2    3. Cat?
        3       None
        dtype: object
        >>> s.str.strip('123.!? \\n\\t')
        0     Ant
        1     Bee
        2     Cat
        3    None
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_strip(self._column, as_scalar(to_strip)), **kwargs
        )

    def lstrip(self, to_strip=None, **kwargs):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from
        each string in the Series/Index from left side.
        Equivalent to `str.lstrip()
        <https://docs.python.org/3/library/stdtypes.html#str.lstrip>`_.

        Parameters
        ----------
        to_strip : str or None, default None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will
            be stripped. If None then whitespaces are removed.

        Returns
        -------
            Series or Index of object

        See also
        --------
        strip
            Remove leading and trailing characters in Series/Index.

        rstrip
            Remove trailing characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\\n', '3. Cat?\\t', None])
        >>> s.str.lstrip('123.')
        0     Ant.
        1     Bee!\\n
        2     Cat?\\t
        3       None
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_lstrip(self._column, as_scalar(to_strip)), **kwargs
        )

    def rstrip(self, to_strip=None, **kwargs):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines)
        or a set of specified characters from each
        string in the Series/Index from right side.
        Equivalent to `str.rstrip()
        <https://docs.python.org/3/library/stdtypes.html#str.rstrip>`_.

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

        See also
        --------
        strip
            Remove leading and trailing characters in Series/Index.

        lstrip
            Remove leading characters in Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['1. Ant.  ', '2. Bee!\\n', '3. Cat?\\t', None])
        >>> s
        0    1. Ant.
        1    2. Bee!\\n
        2    3. Cat?\\t
        3         None
        dtype: object
        >>> s.str.rstrip('.!? \\n\\t')
        0    1. Ant
        1    2. Bee
        2    3. Cat
        3      None
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_rstrip(self._column, as_scalar(to_strip)), **kwargs
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
        stringr library ``str_wrap`` function, the equivalent
        pandas implementation can be obtained using the
        following parameter setting:

            expand_tabs = False

            replace_whitespace = True

            drop_whitespace = True

            break_long_words = False

            break_on_hyphens = False

        Examples
        --------
        >>> import cudf
        >>> data = ['line to be wrapped', 'another line to be wrapped']
        >>> s = cudf.Series(data)
        >>> s.str.wrap(12)
        0             line to be\\nwrapped
        1    another line\\nto be\\nwrapped
        dtype: object
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

        Notes
        -----
            -  `flags` parameter is currently not supported.
            -  Some characters need to be escaped when passing
               in pat. eg. ``'$'`` has a special meaning in regex
               and must be escaped when finding this literal character.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A', 'B', 'Aaba', 'Baca', None, 'CABA', 'cat'])
        >>> s.str.count('a')
        0       0
        1       0
        2       2
        3       2
        4    null
        5       0
        6       1
        dtype: int32

        Escape ``'$'`` to find the literal dollar sign.

        >>> s = cudf.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\$')                                       # noqa W605
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        dtype: int32

        This is also available on Index.

        >>> index = cudf.core.index.StringIndex(['A', 'A', 'Aaba', 'cat'])
        >>> index.str.count('a')
        Int64Index([0, 0, 2, 1], dtype='int64')
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

        Notes
        -----
        `flags` parameter is currently not supported.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern ‘Monkey’ returns one match:

        >>> s.str.findall('Monkey')
                0
        0    None
        1  Monkey
        2    None

        When the pattern matches more than one string
        in the Series, all matches are returned:

        >>> s.str.findall('on')
              0
        0    on
        1    on
        2  None

        Regular expressions are supported too. For instance,
        the search for all the strings ending with
        the word ‘on’ is shown next:

        >>> s.str.findall('on$')
              0
        0    on
        1  None
        2  None

        If the pattern is found more than once in the same
        string, then multiple strings are returned as columns:

        >>> s.str.findall('b')
              0     1
        0  None  None
        1  None  None
        2     b     b
        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        kwargs.setdefault("expand", True)
        return self._return_or_inplace(
            cpp_findall(self._column, pat), **kwargs
        )

    def isempty(self, **kwargs):
        """
        Check whether each string is an empty string.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["1", "abc", "", " ", None])
        >>> s.str.isempty()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool
        """
        return self._return_or_inplace(
            (self._parent == "").fillna(False), **kwargs
        )

    def isspace(self, **kwargs):
        """
        Check whether all characters in each string are whitespace.

        This is equivalent to running the Python string method
        `str.isspace()
        <https://docs.python.org/3/library/stdtypes.html#str.isspace>`_
        for each element of the Series/Index.
        If a string has zero characters, False is returned
        for that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same length as
            the original Series/Index.

        See also
        --------
        isalnum
            Check whether all characters are alphanumeric.

        isalpha
            Check whether all characters are alphabetic.

        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isinteger
            Check whether all characters are integer.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        islower
            Check whether all characters are lowercase.

        isupper
            Check whether all characters are uppercase.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([' ', '\\t\\r\\n ', ''])
        >>> s.str.isspace()
        0     True
        1     True
        2    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_isspace(self._column), **kwargs)

    def endswith(self, pat, **kwargs):
        """
        Test if the end of each string element matches a pattern.

        Parameters
        ----------
        pat : str or list-like
            If `str` is an `str`, evaluates whether each string of
            series ends with `pat`.
            If `pat` is a list-like, evaluates whether `self[i]`
            ends with `pat[i]`.
            Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the end of each string element.

        Notes
        -----
        `na` parameter is not yet supported, as cudf uses
        native strings instead of Python objects.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['bat', 'bear', 'caT', None])
        >>> s
        0     bat
        1    bear
        2     caT
        3    None
        dtype: object
        >>> s.str.endswith('t')
        0     True
        1    False
        2    False
        3     null
        dtype: bool
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
        elif is_scalar(pat):
            result_col = cpp_endswith(self._column, as_scalar(pat, "str"))
        else:
            result_col = cpp_endswith_multiple(
                self._column, column.as_column(pat, dtype="str")
            )

        return self._return_or_inplace(result_col, **kwargs)

    def startswith(self, pat, **kwargs):
        """
        Test if the start of each string element matches a pattern.

        Equivalent to `str.startswith()
        <https://docs.python.org/3/library/stdtypes.html#str.startswith>`_.

        Parameters
        ----------
        pat : str or list-like
            If `str` is an `str`, evaluates whether each string of
            series starts with `pat`.
            If `pat` is a list-like, evaluates whether `self[i]`
            starts with `pat[i]`.
            Regular expressions are not accepted.

        Returns
        -------
        Series or Index of bool
            A Series of booleans indicating whether the given
            pattern matches the start of each string element.

        See also
        --------
        endswith
            Same as startswith, but tests the end of string.

        contains
            Tests if string element contains a pattern.

        Examples
        --------
        >>> import cudf
        >>> s
        0     bat
        1    Bear
        2     cat
        3    None
        dtype: object
        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3     null
        dtype: bool
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
        elif is_scalar(pat):
            result_col = cpp_startswith(self._column, as_scalar(pat, "str"))
        else:
            result_col = cpp_startswith_multiple(
                self._column, column.as_column(pat, dtype="str")
            )

        return self._return_or_inplace(result_col, **kwargs)

    def find(self, sub, start=0, end=None, **kwargs):
        """
        Return lowest indexes in each strings in the Series/Index
        where the substring is fully contained between ``[start:end]``.
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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.find('b')
        0    1
        1   -1
        2    0
        3    2
        dtype: int32

        Parameters such as `start` and `end` can also be used.

        >>> s.str.find('b', start=1, end=5)
        0    1
        1   -1
        2   -1
        3    2
        dtype: int32
        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        if end is None:
            end = -1

        result_col = cpp_find(self._column, as_scalar(sub, "str"), start, end)

        return self._return_or_inplace(result_col, **kwargs)

    def rfind(self, sub, start=0, end=None, **kwargs):
        """
        Return highest indexes in each strings in the Series/Index
        where the substring is fully contained between ``[start:end]``.
        Return -1 on failure. Equivalent to standard `str.rfind()
        <https://docs.python.org/3/library/stdtypes.html#str.rfind>`_.

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

        See also
        --------
        find
            Return lowest indexes in each strings.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["abc", "hello world", "rapids ai"])
        >>> s.str.rfind('a')
        0    0
        1   -1
        2    7
        dtype: int32

        Using `start` and `end` parameters.

        >>> s.str.rfind('a', start=2, end=5)
        0   -1
        1   -1
        2   -1
        dtype: int32
        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        if end is None:
            end = -1

        result_col = cpp_rfind(self._column, as_scalar(sub, "str"), start, end)

        return self._return_or_inplace(result_col, **kwargs)

    def index(self, sub, start=0, end=None, **kwargs):
        """
        Return lowest indexes in each strings where the substring
        is fully contained between ``[start:end]``. This is the same
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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.index('b')
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        ValueError: substring not found

        Parameters such as `start` and `end` can also be used.

        >>> s = cudf.Series(['abc', 'abb','ab' ,'ddb'])
        >>> s.str.index('b', start=1, end=5)
        0    1
        1    1
        2    1
        3    2
        dtype: int32
        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        if end is None:
            end = -1

        result_col = cpp_find(self._column, as_scalar(sub, "str"), start, end)

        result = self._return_or_inplace(result_col, **kwargs)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result

    def rindex(self, sub, start=0, end=None, **kwargs):
        """
        Return highest indexes in each strings where the substring
        is fully contained between ``[start:end]``. This is the same
        as ``str.rfind`` except instead of returning -1, it raises a
        ``ValueError`` when the substring is not found.

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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['abc', 'a','b' ,'ddb'])
        >>> s.str.rindex('b')
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        ValueError: substring not found

        Parameters such as `start` and `end` can also be used.

        >>> s = cudf.Series(['abc', 'abb','ab' ,'ddb'])
        >>> s.str.rindex('b', start=1, end=5)
        0    1
        1    2
        2    1
        3    2
        dtype: int32
        """
        if not isinstance(sub, str):
            msg = "expected a string object, not {0}"
            raise TypeError(msg.format(type(sub).__name__))

        if end is None:
            end = -1

        result_col = cpp_rfind(self._column, as_scalar(sub, "str"), start, end)

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

        Notes
        -----
        Parameters currently not supported are: `case`, `flags` and `na`.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["rapids", "ai", "cudf"])

        Checking for strings starting with `a`.

        >>> s.str.match('a')
        0    False
        1     True
        2    False
        dtype: bool

        Checking for strings starting with any of `a` or `c`.

        >>> s.str.match('[ac]')
        0    False
        1     True
        2     True
        dtype: bool
        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        if flags != 0:
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

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A%2FB-C%2FD', 'e%20f.g', '4-5%2C6'])
        >>> s.str.url_decode()
        0    A/B-C/D
        1      e f.g
        2      4-5,6
        dtype: object
        >>> data = ["https%3A%2F%2Frapids.ai%2Fstart.html", "https%3A%2F%2Fmedium.com%2Frapids-ai"]     # noqa E501
        >>> s = cudf.Series(data)
        >>> s.str.url_decode()
        0    https://rapids.ai/start.html
        1    https://medium.com/rapids-ai
        dtype: object
        """

        return self._return_or_inplace(cpp_url_decode(self._column), **kwargs)

    def url_encode(self, **kwargs):
        """
        Returns a URL-encoded format of each string.
        No format checking is performed.
        All characters are encoded except for ASCII letters,
        digits, and these characters: ``‘.’,’_’,’-‘,’~’``.
        Encoding converts to hex using UTF-8 encoded bytes.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['A/B-C/D', 'e f.g', '4-5,6'])
        >>> s.str.url_encode()
        0    A%2FB-C%2FD
        1        e%20f.g
        2        4-5%2C6
        dtype: object
        >>> data = ["https://rapids.ai/start.html", "https://medium.com/rapids-ai"]         # noqa E501
        >>> s = cudf.Series(data)
        >>> s.str.url_encode()
        0    https%3A%2F%2Frapids.ai%2Fstart.html
        1    https%3A%2F%2Fmedium.com%2Frapids-ai
        dtype: object
        """
        return self._return_or_inplace(cpp_url_encode(self._column), **kwargs)

    def code_points(self, **kwargs):
        """
        Returns an array by filling it with the UTF-8 code point
        values for each character of each string.
        This function uses the ``len()`` method to determine
        the size of each sub-array of integers.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["a","xyz", "éee"])
        >>> s.str.code_points()
        0       97
        1      120
        2      121
        3      122
        4    50089
        5      101
        6      101
        dtype: int32
        >>> s = cudf.Series(["abc"])
        >>> s.str.code_points()
        0    97
        1    98
        2    99
        dtype: int32
        """

        new_col = cpp_code_points(self._column)
        if self._parent is None:
            return new_col
        elif isinstance(self._parent, cudf.Series):
            return cudf.Series(new_col, name=self._parent.name)
        elif isinstance(self._parent, cudf.Index):
            return cudf.core.index.as_index(new_col, name=self._parent.name)

    def translate(self, table, **kwargs):
        """
        Map all characters in the string through the given
        mapping table.

        Equivalent to standard `str.translate()
        <https://docs.python.org/3/library/stdtypes.html#str.translate>`_.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode
            ordinals, strings, or None.
            Unmapped characters are left untouched.
            `str.maketrans()
            <https://docs.python.org/3/library/stdtypes.html#str.maketrans>`_
            is a helper function for making translation tables.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> data = ['lower', 'CAPITALS', 'this is a sentence','SwApCaSe']
        >>> s = cudf.Series(data)
        >>> s.str.translate({'a': "1"})
        0                 lower
        1              CAPITALS
        2    this is 1 sentence
        3              SwApC1Se
        dtype: object
        >>> s.str.translate({'a': "1", "e":"#"})
        0                 low#r
        1              CAPITALS
        2    this is 1 s#nt#nc#
        3              SwApC1S#
        dtype: object
        """
        table = str.maketrans(table)
        return self._return_or_inplace(
            cpp_translate(self._column, table), **kwargs
        )

    def normalize_spaces(self, **kwargs):
        """
        Remove extra whitespace between tokens and trim whitespace
        from the beginning and the end of each string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello \\t world"," test string  "])
        >>> ser.str.normalize_spaces()
        0    hello world
        1    test string
        dtype: object
        """
        return self._return_or_inplace(
            cpp_normalize_spaces(self._column), **kwargs
        )

    def tokenize(self, delimiter=" ", **kwargs):
        """
        Each string is split into tokens using the provided delimiter(s).
        The sequence returned contains the tokens in the order
        they were found.

        Parameters
        ----------
        delimiter : str or list of strs, Default is whitespace.
            The string used to locate the split points of each string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> data = ["hello world", "goodbye world", "hello goodbye"]
        >>> ser = cudf.Series(data)
        >>> ser.str.tokenize()
        0      hello
        1      world
        2    goodbye
        3      world
        4      hello
        5    goodbye
        dtype: object
        """
        delimiter = _massage_string_arg(delimiter, "delimiter", allow_col=True)
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_tokenize(self._column, delimiter), **kwargs
        )

    def character_tokenize(self, **kwargs):
        """
        Each string is split into individual characters.
        The sequence returned contains each character as an individual string.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> data = ["hello world", None, "goodbye, thank you."]
        >>> ser = cudf.Series(data)
        >>> ser.str.character_tokenize()
        0     h
        1     e
        2     l
        3     l
        4     o
        5
        6     w
        7     o
        8     r
        9     l
        10    d
        11    g
        12    o
        13    o
        14    d
        15    b
        16    y
        17    e
        18    ,
        19
        20    t
        21    h
        22    a
        23    n
        24    k
        25
        26    y
        27    o
        28    u
        29    .
        dtype: object
        """

        result_col = cpp_character_tokenize(self._column)
        if self._parent is None:
            return result_col
        elif isinstance(self._parent, cudf.Series):
            return cudf.Series(result_col, name=self._parent.name)
        elif isinstance(self._parent, cudf.Index):
            return cudf.core.index.as_index(result_col, name=self._parent.name)

    def token_count(self, delimiter=" ", **kwargs):
        """
        Each string is split into tokens using the provided delimiter.
        The returned integer sequence is the number of tokens in each string.

        Parameters
        ----------

        delimiter : str or list of strs, Default is whitespace.
            The characters or strings used to locate the
            split points of each string.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello world","goodbye",""])
        >>> ser.str.token_count()
        0    2
        1    1
        2    0
        dtype: int32
        """
        delimiter = _massage_string_arg(delimiter, "delimiter", allow_col=True)
        return self._return_or_inplace(
            cpp_count_tokens(self._column, delimiter), **kwargs
        )

    def ngrams(self, n=2, separator="_", **kwargs):
        """
        Generate the n-grams from a set of tokens, each record
        in series is treated a token.

        You can generate tokens from a Series instance using
        the ``Series.str.tokenize()`` function.

        Parameters
        ----------
        n : int
            The degree of the n-gram (number of consecutive tokens).
            Default of 2 for bigrams.
        separator : str
            The separator to use between within an n-gram.
            Default is '_'.

        Examples
        --------
        >>> import cudf
        >>> str_series = cudf.Series(['this is my', 'favorite book'])
        >>> str_series = cudf.Series(['this is my', 'favorite book'])
        >>> str_series.str.ngrams(2, "_")
        0    this is my_favorite book
        dtype: object
        >>> str_series = cudf.Series(['abc','def','xyz','hhh'])
        >>> str_series.str.ngrams(2, "_")
        0    abc_def
        1    def_xyz
        2    xyz_hhh
        dtype: object
        """
        separator = _massage_string_arg(separator, "separator")
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_generate_ngrams(self._column, n, separator), **kwargs
        )

    def character_ngrams(self, n=2, **kwargs):
        """
        Generate the n-grams from characters in a column of strings.

        Parameters
        ----------
        n : int
            The degree of the n-gram (number of consecutive characters).
            Default of 2 for bigrams.

        Examples
        --------
        >>> import cudf
        >>> str_series = cudf.Series(['abcd','efgh','xyz'])
        >>> str_series.str.character_ngrams(2)
        0    ab
        1    bc
        2    cd
        3    ef
        4    fg
        5    gh
        6    xy
        7    yz
        dtype: object
        >>> str_series.str.character_ngrams(3)
        0    abc
        1    bcd
        2    efg
        3    fgh
        4    xyz
        dtype: object
        """
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_generate_character_ngrams(self._column, n), **kwargs
        )

    def ngrams_tokenize(self, n=2, delimiter=" ", separator="_", **kwargs):
        """
        Generate the n-grams using tokens from each string.
        This will tokenize each string and then generate ngrams for each
        string.

        Parameters
        ----------
        n : int, Default 2.
            The degree of the n-gram (number of consecutive tokens).
        delimiter : str, Default is white-space.
            The character used to locate the split points of each string.
        sep : str, Default is '_'.
            The separator to use between tokens within an n-gram.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(['this is the', 'best book'])
        >>> ser.str.ngrams_tokenize(n=2, sep='_')
        0      this_is
        1       is_the
        2    best_book
        dtype: object
        """
        delimiter = _massage_string_arg(delimiter, "delimiter")
        separator = _massage_string_arg(separator, "separator")
        kwargs.setdefault("retain_index", False)
        return self._return_or_inplace(
            cpp_ngrams_tokenize(self._column, n, delimiter, separator),
            **kwargs,
        )

    def replace_tokens(self, targets, replacements, delimiter=None, **kwargs):
        """
        The targets tokens are searched for within each string in the series
        and replaced with the corresponding replacements if found.
        Tokens are identified by the delimiter character provided.

        Parameters
        ----------
        targets : array-like, Sequence or Series
            The tokens to search for inside each string.

        replacements : array-like, Sequence, Series or str
            The strings to replace for each found target token found.
            Alternately, this can be a single str instance and would be
            used as replacement for each string found.

        delimiter : str
            The character used to locate the tokens of each string.
            Default is whitespace.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["this is me", "theme music", ""])
        >>> targets = cudf.Series(["is", "me"])
        >>> sr.str.replace_tokens(targets=targets, replacements="_")
        0       this _ _
        1    theme music
        2
        dtype: object
        >>> sr = cudf.Series(["this;is;me", "theme;music", ""])
        >>> sr.str.replace_tokens(targets=targets, replacements=":")
        0     this;is;me
        1    theme;music
        2
        dtype: object
        """
        if can_convert_to_column(targets):
            targets_column = column.as_column(targets)
        else:
            raise TypeError(
                f"targets should be an array-like or a Series object, "
                f"found {type(targets)}"
            )

        if is_scalar(replacements):
            replacements_column = column.as_column([replacements])
        elif can_convert_to_column(replacements):
            replacements_column = column.as_column(replacements)
            if len(targets_column) != len(replacements_column):
                raise ValueError(
                    "targets and replacements should be same size"
                    " sequences unless replacements is a string."
                )
        else:
            raise TypeError(
                f"replacements should be an str, array-like or Series object, "
                f"found {type(replacements)}"
            )

        if delimiter is None:
            delimiter = ""
        elif not is_scalar(delimiter):
            raise TypeError(
                f"Type of delimiter should be a string,"
                f" found {type(delimiter)}"
            )

        return self._return_or_inplace(
            cpp_replace_tokens(
                self._column,
                targets_column,
                replacements_column,
                as_scalar(delimiter, dtype="str"),
            ),
            **kwargs,
        )


def _massage_string_arg(value, name, allow_col=False):
    if isinstance(value, str):
        return as_scalar(value, dtype="str")

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

    def set_base_children(self, value):
        # TODO: Implement dtype validation of the children here somehow
        super().set_base_children(value)

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
                chars_offset = libcudf.copying.get_element(offsets_column, 0)
                offsets_column = offsets_column.binary_operator(
                    "sub", chars_offset
                )

                # Shift the chars offset by the new first element of the
                # offsets column
                chars_size = libcudf.copying.get_element(
                    offsets_column, self.size
                )

                chars_column = column.build_column(
                    data=chars_column.base_data,
                    dtype=chars_column.dtype,
                    mask=chars_column.base_mask,
                    size=chars_size.value,
                    offset=chars_offset.value,
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

    def _memory_usage(self, **kwargs):
        return self.__sizeof__()

    def unary_operator(self, unaryop):
        raise TypeError(
            f"Series of dtype `str` cannot perform the operation: "
            f"{unaryop}"
        )

    def __len__(self):
        return self.size

    def _set_mask(self, value):
        super()._set_mask(value)

    @property
    def _nbytes(self):
        if self.size == 0:
            return 0
        else:
            return self.children[1].size

    def as_numerical_column(self, dtype, **kwargs):

        out_dtype = np.dtype(dtype)
        kwargs.update(dtype=out_dtype)

        if out_dtype.type is np.datetime64:
            if "format" not in kwargs:
                if len(self) > 0:
                    # infer on host from the first not na element
                    fmt = datetime.infer_format(self[self.notna()][0])
                    kwargs.update(format=fmt)

            # Check for None strings
            if len(self) > 0 and self.binary_operator("eq", "None").any():
                raise ValueError("Could not convert `None` value to datetime")

            boolean_match = self.binary_operator("eq", "NaT")
        elif out_dtype.kind in {"i", "u"}:
            if not cpp_is_integer(self).all():
                raise ValueError(
                    "Could not convert strings to integer \
                        type due to presence of non-integer values."
                )
        elif out_dtype.kind == "f":
            if not cpp_is_float(self).all():
                raise ValueError(
                    "Could not convert strings to float \
                        type due to presence of non-floating values."
                )

        result_col = _str_to_numeric_typecast_functions[out_dtype](
            self, **kwargs
        )
        if (out_dtype.type is np.datetime64) and boolean_match.any():
            result_col[boolean_match] = None
        return result_col

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

    def can_cast_safely(self, to_dtype):
        to_dtype = np.dtype(to_dtype)

        if self.dtype == to_dtype:
            return True
        elif to_dtype.kind in {"i", "u"} and not cpp_is_integer(self).all():
            return False
        elif to_dtype.kind == "f" and not cpp_is_float(self).all():
            return False
        else:
            return True

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
        return out


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def _string_column_binop(lhs, rhs, op, out_dtype):
    out = libcudf.binaryop.binaryop(lhs=lhs, rhs=rhs, op=op, dtype=out_dtype)
    return out


def _get_cols_list(others):

    if (
        can_convert_to_column(others)
        and len(others) > 0
        and (
            can_convert_to_column(
                others.iloc[0]
                if isinstance(others, cudf.Series)
                else others[0]
            )
        )
    ):
        """
        If others is a list-like object (in our case lists & tuples)
        just another Series/Index, great go ahead with concatenation.
        """
        cols_list = [column.as_column(frame, dtype="str") for frame in others]
        return cols_list
    elif others is not None:
        return [column.as_column(others, dtype="str")]
    else:
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarrary "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )
