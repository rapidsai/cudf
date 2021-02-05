# Copyright (c) 2019-2021, NVIDIA CORPORATION.

from __future__ import annotations

import builtins
import pickle
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast, overload

import cupy
import numpy as np
import pandas as pd
from numba import cuda
from nvtx import annotate

import cudf
from cudf import _lib as libcudf
from cudf._lib import string_casting as str_cast
from cudf._lib.column import Column
from cudf._lib.nvtext.edit_distance import edit_distance as cpp_edit_distance
from cudf._lib.nvtext.generate_ngrams import (
    generate_character_ngrams as cpp_generate_character_ngrams,
    generate_ngrams as cpp_generate_ngrams,
)
from cudf._lib.nvtext.ngrams_tokenize import (
    ngrams_tokenize as cpp_ngrams_tokenize,
)
from cudf._lib.nvtext.normalize import (
    normalize_characters as cpp_normalize_characters,
    normalize_spaces as cpp_normalize_spaces,
)
from cudf._lib.nvtext.replace import (
    filter_tokens as cpp_filter_tokens,
    replace_tokens as cpp_replace_tokens,
)
from cudf._lib.nvtext.stemmer import (
    LetterType,
    is_letter as cpp_is_letter,
    is_letter_multi as cpp_is_letter_multi,
    porter_stemmer_measure as cpp_porter_stemmer_measure,
)
from cudf._lib.nvtext.subword_tokenize import (
    subword_tokenize as cpp_subword_tokenize,
)
from cudf._lib.nvtext.tokenize import (
    _count_tokens_column as cpp_count_tokens_column,
    _count_tokens_scalar as cpp_count_tokens_scalar,
    _tokenize_column as cpp_tokenize_column,
    _tokenize_scalar as cpp_tokenize_scalar,
    character_tokenize as cpp_character_tokenize,
    detokenize as cpp_detokenize,
)
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
    filter_alphanum as cpp_filter_alphanum,
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
    contains_multiple as cpp_contains_multiple,
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
    rsplit_record as cpp_rsplit_record,
    split as cpp_split,
    split_record as cpp_split_record,
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
from cudf._lib.strings.translate import (
    filter_characters as cpp_filter_characters,
    translate as cpp_translate,
)
from cudf._lib.strings.wrap import wrap as cpp_wrap
from cudf._typing import ColumnLike, Dtype, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import column, datetime
from cudf.core.column.methods import ColumnMethodsMixin
from cudf.utils import utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    is_list_dtype,
    is_scalar,
    is_string_dtype,
)

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
}

_datetime_to_str_typecast_functions = {
    # TODO: support Date32 UNIX days
    # np.dtype("datetime64[D]"): str_cast.int2timestamp,
    np.dtype("datetime64[s]"): str_cast.int2timestamp,
    np.dtype("datetime64[ms]"): str_cast.int2timestamp,
    np.dtype("datetime64[us]"): str_cast.int2timestamp,
    np.dtype("datetime64[ns]"): str_cast.int2timestamp,
}

_timedelta_to_str_typecast_functions = {
    np.dtype("timedelta64[s]"): str_cast.int2timedelta,
    np.dtype("timedelta64[ms]"): str_cast.int2timedelta,
    np.dtype("timedelta64[us]"): str_cast.int2timedelta,
    np.dtype("timedelta64[ns]"): str_cast.int2timedelta,
}


ParentType = Union["cudf.Series", "cudf.Index"]


class StringMethods(ColumnMethodsMixin):
    def __init__(self, column, parent=None):
        """
        Vectorized string functions for Series and Index.

        This mimics pandas ``df.str`` interface. nulls stay null
        unless handled otherwise by a particular method.
        Patterned after Python’s string methods, with some
        inspiration from R’s stringr package.
        """
        value_type = (
            column.dtype.leaf_type if is_list_dtype(column) else column.dtype
        )
        if not is_string_dtype(value_type):
            raise AttributeError(
                "Can only use .str accessor with string values"
            )
        super().__init__(column=column, parent=parent)

    def htoi(self) -> ParentType:
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

    def ip2int(self) -> ParentType:
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

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def len(self) -> ParentType:
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
        3    <NA>
        dtype: int32
        """

        return self._return_or_inplace(cpp_count_characters(self._column))

    def byte_count(self) -> ParentType:
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
        return self._return_or_inplace(cpp_count_bytes(self._column),)

    @overload
    def cat(self, sep: str = None, na_rep: str = None) -> str:
        ...

    @overload
    def cat(
        self, others, sep: str = None, na_rep: str = None
    ) -> Union[ParentType, "cudf.core.column.StringColumn"]:
        ...

    def cat(self, others=None, sep=None, na_rep=None):
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
        2    <NA>
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
                self._column, cudf.Scalar(sep), cudf.Scalar(na_rep, "str"),
            )
        else:
            other_cols = _get_cols_list(self._parent, others)
            all_cols = [self._column] + other_cols
            data = cpp_concatenate(
                cudf.DataFrame(
                    {index: value for index, value in enumerate(all_cols)}
                ),
                cudf.Scalar(sep),
                cudf.Scalar(na_rep, "str"),
            )

        if len(data) == 1 and data.null_count == 1:
            data = [""]
        out = self._return_or_inplace(data)
        if len(out) == 1 and others is None:
            if isinstance(out, cudf.Series):
                out = out.iloc[0]
            else:
                out = out[0]
        return out

    def join(self, sep) -> ParentType:
        """
        Join lists contained as elements in the Series/Index with passed
        delimiter.

        Raises : NotImplementedError
            Columns of arrays / lists are not yet supported.
        """
        raise NotImplementedError(
            "Columns of arrays / lists are not yet " "supported"
        )

    def extract(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> ParentType:
        """
        Extract capture groups in the regex `pat` as columns in a DataFrame.

        For each subject string in the Series, extract groups from the first
        match of regular expression `pat`.

        Parameters
        ----------
        pat : str
            Regular expression pattern with capturing groups.
        expand : bool, default True
            If True, return DataFrame with one column per capture group.
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
        2  <NA>  <NA>

        A pattern with one group will return a DataFrame with one
        column if expand=True.

        >>> s.str.extract(r'[ab](\d)', expand=True)                     # noqa W605
              0
        0     1
        1     2
        2  <NA>

        A pattern with one group will return a Series if expand=False.

        >>> s.str.extract(r'[ab](\d)', expand=False)                    # noqa W605
        0       1
        1       2
        2    <NA>
        dtype: object
        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        out = cpp_extract(self._column, pat)
        if out._num_columns == 1 and expand is False:
            return self._return_or_inplace(out._columns[0], expand=expand)
        else:
            return self._return_or_inplace(out, expand=expand)

    def contains(
        self,
        pat: Union[str, Sequence],
        case: bool = True,
        flags: int = 0,
        na=np.nan,
        regex: bool = True,
    ) -> ParentType:
        """
        Test if pattern or regex is contained within a string of a Series or
        Index.

        Return boolean Series or Index based on whether a given pattern or
        regex is contained within a string of a Series or Index.

        Parameters
        ----------
        pat : str or list-like
            Character sequence or regular expression.
            If ``pat`` is list-like then regular expressions are not
            accepted.
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
        4                <NA>
        dtype: object
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4     <NA>
        dtype: bool

        Returning an Index of booleans using only a literal pattern.

        >>> data = ['Mouse', 'dog', 'house and parrot', '23.0', np.NaN]
        >>> idx = cudf.Index(data)
        >>> idx
        StringIndex(['Mouse' 'dog' 'house and parrot' '23.0' None], dtype='object')
        >>> idx.str.contains('23', regex=False)
        GenericIndex([False, False, False, True, <NA>], dtype='bool')

        Returning ‘house’ or ‘dog’ when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4     <NA>
        dtype: bool

        Returning any digit using regular expression.

        >>> s1.str.contains('\d', regex=True)                               # noqa W605
        0    False
        1    False
        2    False
        3     True
        4     <NA>
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

        The ``pat`` may also be a list of strings in which case
        the individual strings are searched in corresponding rows.

        >>> s2 = cudf.Series(['house', 'dog', 'and', '', ''])
        >>> s1.str.contains(s2)
        0    False
        1     True
        2     True
        3     True
        4     <NA>
        dtype: bool
        """
        if case is not True:
            raise NotImplementedError("`case` parameter is not yet supported")
        elif flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")
        elif na is not np.nan:
            raise NotImplementedError("`na` parameter is not yet supported")

        if pat is None:
            result_col = column.column_empty(
                len(self._column), dtype="bool", masked=True
            )
        elif is_scalar(pat):
            if regex is True:
                result_col = cpp_contains_re(self._column, pat)
            else:
                result_col = cpp_contains(
                    self._column, cudf.Scalar(pat, "str")
                )
        else:
            result_col = cpp_contains_multiple(
                self._column, column.as_column(pat, dtype="str")
            )
        return self._return_or_inplace(result_col)

    def replace(
        self,
        pat: Union[str, Sequence],
        repl: Union[str, Sequence],
        n: int = -1,
        case=None,
        flags: int = 0,
        regex: bool = True,
    ) -> ParentType:
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
        2    <NA>
        dtype: object

        When pat is a string and regex is True (the default), the given pat
        is compiled as a regex. When repl is a string, it replaces matching
        regex patterns as with ``re.sub()``. NaN value(s) in the Series
        are left as is:

        >>> s.str.replace('f.', 'ba', regex=True)
        0     bao
        1     baz
        2    <NA>
        dtype: object

        When pat is a string and `regex` is False, every pat is replaced
        with repl as with ``str.replace()``:

        >>> s.str.replace('f.', 'ba', regex=False)
        0     foo
        1     fuz
        2    <NA>
        dtype: object
        """
        if case is not None:
            raise NotImplementedError("`case` parameter is not yet supported")
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        if can_convert_to_column(pat) and can_convert_to_column(repl):
            warnings.warn(
                "`n` parameter is not supported when "
                "`pat` and `repl` are list-like inputs"
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
            )
        # Pandas treats 0 as all
        if n == 0:
            n = -1

        # Pandas forces non-regex replace when pat is a single-character
        return self._return_or_inplace(
            cpp_replace_re(self._column, pat, cudf.Scalar(repl, "str"), n)
            if regex is True and len(pat) > 1
            else cpp_replace(
                self._column,
                cudf.Scalar(pat, "str"),
                cudf.Scalar(repl, "str"),
                n,
            ),
        )

    def replace_with_backrefs(self, pat: str, repl: str) -> ParentType:
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
        >>> s.str.replace_with_backrefs('(\\\\d)(\\\\d)', 'V\\\\2\\\\1')
        0    AV453
        1    ZV576
        dtype: object
        """
        return self._return_or_inplace(
            cpp_replace_with_backrefs(self._column, pat, repl)
        )

    def slice(
        self, start: int = None, stop: int = None, step: int = None
    ) -> ParentType:
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
            cpp_slice_strings(self._column, start, stop, step),
        )

    def isinteger(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_integer(self._column))

    def ishex(self) -> ParentType:
        """
        Check whether all characters in each string form a hex integer.

        If a string has zero characters, False is returned for
        that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        See also
        --------
        isdecimal
            Check whether all characters are decimal.

        isdigit
            Check whether all characters are digits.

        isnumeric
            Check whether all characters are numeric.

        isfloat
            Check whether all characters are float.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["", "123DEF", "0x2D3", "-15", "abc"])
        >>> s.str.ishex()
        0    False
        1     True
        2     True
        3    False
        4     True
        dtype: bool
        """
        return self._return_or_inplace(str_cast.is_hex(self._column))

    def istimestamp(self, format: str) -> ParentType:
        """
        Check whether all characters in each string can be converted to
        a timestamp using the given format.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["20201101", "192011", "18200111", "2120-11-01"])
        >>> s.str.istimestamp("%Y%m%d")
        0     True
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(
            str_cast.istimestamp(self._column, format)
        )

    def isfloat(self) -> ParentType:
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
        >>> s = cudf.Series(["this is plain text", "\\t\\n", "9.9", "9.9.9"])
        >>> s.str.isfloat()
        0    False
        1    False
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(cpp_is_float(self._column))

    def isdecimal(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_decimal(self._column))

    def isalnum(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_alnum(self._column))

    def isalpha(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_alpha(self._column))

    def isdigit(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_digit(self._column))

    def isnumeric(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_numeric(self._column))

    def isupper(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_upper(self._column))

    def islower(self) -> ParentType:
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
        return self._return_or_inplace(cpp_is_lower(self._column))

    def isipv4(self) -> ParentType:
        """
        Check whether all characters in each string form an IPv4 address.

        If a string has zero characters, False is returned for
        that check.

        Returns : Series or Index of bool
            Series or Index of boolean values with the same
            length as the original Series/Index.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["", "127.0.0.1", "255.255.255.255", "123.456"])
        >>> s.str.isipv4()
        0    False
        1     True
        2     True
        3    False
        dtype: bool
        """
        return self._return_or_inplace(str_cast.is_ipv4(self._column))

    def lower(self) -> ParentType:
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
        return self._return_or_inplace(cpp_to_lower(self._column))

    def upper(self) -> ParentType:
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
        return self._return_or_inplace(cpp_to_upper(self._column))

    def capitalize(self) -> ParentType:
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
        return self._return_or_inplace(cpp_capitalize(self._column))

    def swapcase(self) -> ParentType:
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
        return self._return_or_inplace(cpp_swapcase(self._column))

    def title(self) -> ParentType:
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
        return self._return_or_inplace(cpp_title(self._column))

    def filter_alphanum(
        self, repl: str = None, keep: bool = True
    ) -> ParentType:
        """
        Remove non-alphanumeric characters from strings in this column.

        Parameters
        ----------
        repl : str
            Optional string to use in place of removed characters.
        keep : bool
            Set to False to remove all alphanumeric characters instead
            of keeping them.

        Returns
        -------
        Series/Index of str dtype
            Strings with only alphanumeric characters.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(["pears £12", "plums $34", "Temp 72℉", "100K℧"])
        >>> s.str.filter_alphanum(" ")
        0    pears  12
        1    plums  34
        2     Temp 72
        3        100K
        dtype: object
        """
        if repl is None:
            repl = ""

        return self._return_or_inplace(
            cpp_filter_alphanum(self._column, cudf.Scalar(repl), keep),
        )

    def slice_from(
        self, starts: "cudf.Series", stops: "cudf.Series"
    ) -> ParentType:
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
        )

    def slice_replace(
        self, start: int = None, stop: int = None, repl: str = None
    ) -> ParentType:
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
            cpp_slice_replace(self._column, start, stop, cudf.Scalar(repl)),
        )

    def insert(self, start: int = 0, repl: str = None) -> ParentType:
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
            cpp_string_insert(self._column, start, cudf.Scalar(repl)),
        )

    def get(self, i: int = 0) -> ParentType:
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

        return self._return_or_inplace(cpp_string_get(self._column, i))

    def split(
        self, pat: str = None, n: int = -1, expand: bool = None
    ) -> ParentType:
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
        expand : bool, default False
            Expand the split strings into separate columns.

            * If ``True``, return DataFrame/MultiIndex expanding
              dimensionality.
            * If ``False``, return Series/Index, containing lists
              of strings.

        Returns
        -------
        Series, Index, DataFrame or MultiIndex
            Type matches caller unless ``expand=True`` (see Notes).

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
        The handling of the n keyword depends on the number
        of found splits:

            - If found splits > n, make first n splits only
            - If found splits <= n, make all splits
            - If for a certain row the number of found
              splits < n, append None for padding up to n
              if ``expand=True``.

        If using ``expand=True``, Series and Index callers return
        DataFrame and MultiIndex objects, respectively.

        Examples
        --------
        >>> import cudf
        >>> data = ["this is a regular sentence",
        ...     "https://docs.python.org/index.html", None]
        >>> s = cudf.Series(data)
        >>> s
        0            this is a regular sentence
        1    https://docs.python.org/index.html
        2                                  <NA>
        dtype: object

        In the default setting, the string is split by whitespace.

        >>> s.str.split()
        0        [this, is, a, regular, sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        Without the ``n`` parameter, the outputs of ``rsplit``
        and ``split`` are identical.

        >>> s.str.rsplit()
        0        [this, is, a, regular, sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        The `n` parameter can be used to limit the number of
        splits on the delimiter.

        >>> s.str.split(n=2)
        0          [this, is, a regular sentence]
        1    [https://docs.python.org/index.html]
        2                                    None
        dtype: list

        The `pat` parameter can be used to split by other characters.

        >>> s.str.split(pat="/")
        0               [this is a regular sentence]
        1    [https:, , docs.python.org, index.html]
        2                                       None
        dtype: list

        When using ``expand=True``, the split elements will expand out
        into separate columns. If ``<NA>`` value is present, it is propagated
        throughout the columns during the split.

        >>> s.str.split(expand=True)
                                            0     1     2        3         4
        0                                this    is     a  regular  sentence
        1  https://docs.python.org/index.html  <NA>  <NA>     <NA>      <NA>
        2                                <NA>  <NA>  <NA>     <NA>      <NA>
        """

        if expand is None:
            expand = False

        if expand not in (True, False):
            raise ValueError(
                f"expand parameter accepts only : [True, False], "
                f"got {expand}"
            )

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        if pat is None:
            pat = ""

        if expand:
            if self._column.null_count == len(self._column):
                result_table = cudf.core.frame.Frame({0: self._column.copy()})
            else:
                result_table = cpp_split(
                    self._column, cudf.Scalar(pat, "str"), n
                )
                if len(result_table._data) == 1:
                    if result_table._data[0].null_count == len(self._column):
                        result_table = cudf.core.frame.Frame({})
        else:
            result_table = cpp_split_record(
                self._column, cudf.Scalar(pat, "str"), n
            )

        return self._return_or_inplace(result_table, expand=expand)

    def rsplit(
        self, pat: str = None, n: int = -1, expand: bool = None
    ) -> ParentType:
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
        expand : bool, default False
            Expand the split strings into separate columns.

            * If ``True``, return DataFrame/MultiIndex expanding
              dimensionality.
            * If ``False``, return Series/Index, containing lists
              of strings.

        Returns
        -------
        Series, Index, DataFrame or MultiIndex
            Type matches caller unless ``expand=True`` (see Notes).

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
        The handling of the n keyword depends on the number of
        found splits:

            - If found splits > n, make first n splits only
            - If found splits <= n, make all splits
            - If for a certain row the number of found splits < n,
              append None for padding up to n if ``expand=True``.

        If using ``expand=True``, Series and Index callers return
        DataFrame and MultiIndex objects, respectively.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(
        ...     [
        ...         "this is a regular sentence",
        ...         "https://docs.python.org/3/tutorial/index.html",
        ...         None
        ...     ]
        ... )
        >>> s
        0                       this is a regular sentence
        1    https://docs.python.org/3/tutorial/index.html
        2                                             <NA>
        dtype: object

        In the default setting, the string is split by whitespace.

        >>> s.str.rsplit()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        Without the ``n`` parameter, the outputs of ``rsplit``
        and ``split`` are identical.

        >>> s.str.split()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        The n parameter can be used to limit the number of
        splits on the delimiter. The outputs of split and rsplit are different.

        >>> s.str.rsplit(n=2)
        0                     [this is a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list
        >>> s.str.split(n=2)
        0                     [this, is, a regular sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        dtype: list

        When using ``expand=True``, the split elements will expand
        out into separate columns. If ``<NA>`` value is present,
        it is propagated throughout the columns during the split.

        >>> s.str.rsplit(n=2, expand=True)
                                                       0        1         2
        0                                      this is a  regular  sentence
        1  https://docs.python.org/3/tutorial/index.html     <NA>      <NA>
        2                                           <NA>     <NA>      <NA>

        For slightly more complex use cases like splitting the
        html document name from a url, a combination of parameter
        settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True)
                                            0           1
        0          this is a regular sentence        <NA>
        1  https://docs.python.org/3/tutorial  index.html
        2                                <NA>        <NA>
        """

        if expand is None:
            expand = False

        if expand not in (True, False):
            raise ValueError(
                f"expand parameter accepts only : [True, False], "
                f"got {expand}"
            )

        # Pandas treats 0 as all
        if n == 0:
            n = -1

        if pat is None:
            pat = ""

        if expand:
            if self._column.null_count == len(self._column):
                result_table = cudf.core.frame.Frame({0: self._column.copy()})
            else:
                result_table = cpp_rsplit(self._column, cudf.Scalar(pat), n)
                if len(result_table._data) == 1:
                    if result_table._data[0].null_count == len(self._column):
                        result_table = cudf.core.frame.Frame({})
        else:
            result_table = cpp_rsplit_record(self._column, cudf.Scalar(pat), n)

        return self._return_or_inplace(result_table, expand=expand)

    def partition(self, sep: str = " ", expand: bool = True) -> ParentType:
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
        MultiIndex([('X', ' ', '123'),
                    ('Y', ' ', '999')],
                   )
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        if sep is None:
            sep = " "

        return self._return_or_inplace(
            cpp_partition(self._column, cudf.Scalar(sep)), expand=expand
        )

    def rpartition(self, sep: str = " ", expand: bool = True) -> ParentType:
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
        MultiIndex([('X', ' ', '123'),
                    ('Y', ' ', '999')],
                   )
        """
        if expand is not True:
            raise NotImplementedError(
                "`expand=False` is currently not supported"
            )

        if sep is None:
            sep = " "

        return self._return_or_inplace(
            cpp_rpartition(self._column, cudf.Scalar(sep)), expand=expand
        )

    def pad(
        self, width: int, side: str = "left", fillchar: str = " "
    ) -> ParentType:
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
            cpp_pad(self._column, width, fillchar, side)
        )

    def zfill(self, width: int) -> ParentType:
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
        3    <NA>
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
        3    <NA>
        dtype: object
        """
        if not pd.api.types.is_integer(width):
            msg = f"width must be of integer type, not {type(width).__name__}"
            raise TypeError(msg)

        return self._return_or_inplace(cpp_zfill(self._column, width))

    def center(self, width: int, fillchar: str = " ") -> ParentType:
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
        2    <NA>
        3       d
        dtype: object
        >>> s.str.center(1, fillchar='-')
        0       a
        1       b
        2    <NA>
        3       d
        dtype: object
        >>> s.str.center(2, fillchar='-')
        0      a-
        1      b-
        2    <NA>
        3      d-
        dtype: object
        >>> s.str.center(5, fillchar='-')
        0    --a--
        1    --b--
        2     <NA>
        3    --d--
        dtype: object
        >>> s.str.center(6, fillchar='-')
        0    --a---
        1    --b---
        2      <NA>
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
            cpp_center(self._column, width, fillchar)
        )

    def ljust(self, width: int, fillchar: str = " ") -> ParentType:
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
            cpp_ljust(self._column, width, fillchar)
        )

    def rjust(self, width: int, fillchar: str = " ") -> ParentType:
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
            cpp_rjust(self._column, width, fillchar)
        )

    def strip(self, to_strip: str = None) -> ParentType:
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
        3         <NA>
        dtype: object
        >>> s.str.strip()
        0    1. Ant.
        1    2. Bee!
        2    3. Cat?
        3       <NA>
        dtype: object
        >>> s.str.strip('123.!? \\n\\t')
        0     Ant
        1     Bee
        2     Cat
        3    <NA>
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_strip(self._column, cudf.Scalar(to_strip))
        )

    def lstrip(self, to_strip: str = None) -> ParentType:
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
        3       <NA>
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_lstrip(self._column, cudf.Scalar(to_strip))
        )

    def rstrip(self, to_strip: str = None) -> ParentType:
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
        3         <NA>
        dtype: object
        >>> s.str.rstrip('.!? \\n\\t')
        0    1. Ant
        1    2. Bee
        2    3. Cat
        3      <NA>
        dtype: object
        """
        if to_strip is None:
            to_strip = ""

        return self._return_or_inplace(
            cpp_rstrip(self._column, cudf.Scalar(to_strip))
        )

    def wrap(self, width: int, **kwargs) -> ParentType:
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
                "wrap current implementation defaults to "
                "`break_long_words`=False"
            )

        break_on_hyphens = kwargs.get("break_on_hyphens", None)
        if break_long_words is True:
            raise NotImplementedError(
                "`break_on_hyphens=True` is not supported"
            )
        elif break_on_hyphens is None:
            warnings.warn(
                "wrap current implementation defaults to "
                "`break_on_hyphens`=False"
            )

        return self._return_or_inplace(cpp_wrap(self._column, width))

    def count(self, pat: str, flags: int = 0) -> ParentType:
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
        4    <NA>
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

        return self._return_or_inplace(cpp_count_re(self._column, pat))

    def findall(
        self, pat: str, flags: int = 0, expand: bool = True
    ) -> ParentType:
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
        0    <NA>
        1  Monkey
        2    <NA>

        When the pattern matches more than one string
        in the Series, all matches are returned:

        >>> s.str.findall('on')
              0
        0    on
        1    on
        2  <NA>

        Regular expressions are supported too. For instance,
        the search for all the strings ending with
        the word ‘on’ is shown next:

        >>> s.str.findall('on$')
              0
        0    on
        1  <NA>
        2  <NA>

        If the pattern is found more than once in the same
        string, then multiple strings are returned as columns:

        >>> s.str.findall('b')
              0     1
        0  <NA>  <NA>
        1  <NA>  <NA>
        2     b     b
        """
        if flags != 0:
            raise NotImplementedError("`flags` parameter is not yet supported")

        return self._return_or_inplace(
            cpp_findall(self._column, pat), expand=expand
        )

    def isempty(self) -> ParentType:
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
        return self._return_or_inplace((self._column == "").fillna(False))

    def isspace(self) -> ParentType:
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
        return self._return_or_inplace(cpp_isspace(self._column))

    def endswith(self, pat: str) -> ParentType:
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
        3    <NA>
        dtype: object
        >>> s.str.endswith('t')
        0     True
        1    False
        2    False
        3     <NA>
        dtype: bool
        """
        if pat is None:
            result_col = column.column_empty(
                len(self._column), dtype="bool", masked=True
            )
        elif is_scalar(pat):
            result_col = cpp_endswith(self._column, cudf.Scalar(pat, "str"))
        else:
            result_col = cpp_endswith_multiple(
                self._column, column.as_column(pat, dtype="str")
            )

        return self._return_or_inplace(result_col)

    def startswith(self, pat: Union[str, Sequence]) -> ParentType:
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
        >>> s = cudf.Series(['bat', 'Bear', 'cat', None])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    <NA>
        dtype: object
        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3     <NA>
        dtype: bool
        """
        if pat is None:
            result_col = column.column_empty(
                len(self._column), dtype="bool", masked=True
            )
        elif is_scalar(pat):
            result_col = cpp_startswith(self._column, cudf.Scalar(pat, "str"))
        else:
            result_col = cpp_startswith_multiple(
                self._column, column.as_column(pat, dtype="str")
            )

        return self._return_or_inplace(result_col)

    def find(self, sub: str, start: int = 0, end: int = None) -> ParentType:
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
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = cpp_find(
            self._column, cudf.Scalar(sub, "str"), start, end
        )

        return self._return_or_inplace(result_col)

    def rfind(self, sub: str, start: int = 0, end: int = None) -> ParentType:
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
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = cpp_rfind(
            self._column, cudf.Scalar(sub, "str"), start, end
        )

        return self._return_or_inplace(result_col)

    def index(self, sub: str, start: int = 0, end: int = None) -> ParentType:
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
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = cpp_find(
            self._column, cudf.Scalar(sub, "str"), start, end
        )

        result = self._return_or_inplace(result_col)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result

    def rindex(self, sub: str, start: int = 0, end: int = None) -> ParentType:
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
            raise TypeError(
                f"expected a string object, not {type(sub).__name__}"
            )

        if end is None:
            end = -1

        result_col = cpp_rfind(
            self._column, cudf.Scalar(sub, "str"), start, end
        )

        result = self._return_or_inplace(result_col)

        if (result == -1).any():
            raise ValueError("substring not found")
        else:
            return result

    def match(self, pat: str, case: bool = True, flags: int = 0) -> ParentType:
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

        return self._return_or_inplace(cpp_match_re(self._column, pat))

    def url_decode(self) -> ParentType:
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
        >>> data = ["https%3A%2F%2Frapids.ai%2Fstart.html",
        ...     "https%3A%2F%2Fmedium.com%2Frapids-ai"]
        >>> s = cudf.Series(data)
        >>> s.str.url_decode()
        0    https://rapids.ai/start.html
        1    https://medium.com/rapids-ai
        dtype: object
        """

        return self._return_or_inplace(cpp_url_decode(self._column))

    def url_encode(self) -> ParentType:
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
        >>> data = ["https://rapids.ai/start.html",
        ...     "https://medium.com/rapids-ai"]
        >>> s = cudf.Series(data)
        >>> s.str.url_encode()
        0    https%3A%2F%2Frapids.ai%2Fstart.html
        1    https%3A%2F%2Fmedium.com%2Frapids-ai
        dtype: object
        """
        return self._return_or_inplace(cpp_url_encode(self._column))

    def code_points(self) -> ParentType:
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
        if isinstance(self._parent, cudf.Series):
            return cudf.Series(new_col, name=self._parent.name)
        elif isinstance(self._parent, cudf.Index):
            return cudf.core.index.as_index(new_col, name=self._parent.name)
        else:
            return new_col

    def translate(self, table: dict) -> ParentType:
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
        return self._return_or_inplace(cpp_translate(self._column, table))

    def filter_characters(
        self, table: dict, keep: bool = True, repl: str = None
    ) -> ParentType:
        """
        Remove characters from each string using the character ranges
        in the given mapping table.

        Parameters
        ----------
        table : dict
            This table is a range of Unicode ordinals to filter.
            The minimum value is the key and the maximum value is the value.
            You can use `str.maketrans()
            <https://docs.python.org/3/library/stdtypes.html#str.maketrans>`_
            as a helper function for making the filter table.
            Overlapping ranges will cause undefined results.
            Range values are inclusive.
        keep : boolean
            If False, the character ranges in the ``table`` are removed.
            If True, the character ranges not in the ``table`` are removed.
            Default is True.
        repl : str
            Optional replacement string to use in place of removed characters.

        Returns
        -------
        Series or Index.

        Examples
        --------
        >>> import cudf
        >>> data = ['aeiou', 'AEIOU', '0123456789']
        >>> s = cudf.Series(data)
        >>> s.str.filter_characters({'a':'l', 'M':'Z', '4':'6'})
        0    aei
        1     OU
        2    456
        dtype: object
        >>> s.str.filter_characters({'a':'l', 'M':'Z', '4':'6'}, False, "_")
        0         ___ou
        1         AEI__
        2    0123___789
        dtype: object
        """
        if repl is None:
            repl = ""
        table = str.maketrans(table)
        return self._return_or_inplace(
            cpp_filter_characters(
                self._column, table, keep, cudf.Scalar(repl)
            ),
        )

    def normalize_spaces(self) -> ParentType:
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
        return self._return_or_inplace(cpp_normalize_spaces(self._column))

    def normalize_characters(self, do_lower: bool = True) -> ParentType:
        """
        Normalizes strings characters for tokenizing.

        This uses the normalizer that is built into the
        subword_tokenize function which includes:

            - adding padding around punctuation (unicode category starts with
              "P") as well as certain ASCII symbols like "^" and "$"
            - adding padding around the CJK Unicode block characters
            - changing whitespace (e.g. ``\\t``, ``\\n``, ``\\r``) to space
            - removing control characters (unicode categories "Cc" and "Cf")

        If `do_lower_case = true`, lower-casing also removes the accents.
        The accents cannot be removed from upper-case characters without
        lower-casing and lower-casing cannot be performed without also
        removing accents. However, if the accented character is already
        lower-case, then only the accent is removed.

        Parameters
        ----------
        do_lower : bool, Default is True
            If set to True, characters will be lower-cased and accents
            will be removed. If False, accented and upper-case characters
            are not transformed.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["héllo, \\tworld","ĂĆCĖÑTED","$99"])
        >>> ser.str.normalize_characters()
        0    hello ,  world
        1          accented
        2              $ 99
        dtype: object
        >>> ser.str.normalize_characters(do_lower=False)
        0    héllo ,  world
        1          ĂĆCĖÑTED
        2              $ 99
        dtype: object
        """
        return self._return_or_inplace(
            cpp_normalize_characters(self._column, do_lower)
        )

    def tokenize(self, delimiter: str = " ") -> ParentType:
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

        if isinstance(delimiter, Column):
            return self._return_or_inplace(
                cpp_tokenize_column(self._column, delimiter),
                retain_index=False,
            )
        elif isinstance(delimiter, cudf.Scalar):
            return self._return_or_inplace(
                cpp_tokenize_scalar(self._column, delimiter),
                retain_index=False,
            )
        else:
            raise TypeError(
                f"Expected a Scalar or Column\
                for delimiters, but got {type(delimiter)}"
            )

    def detokenize(
        self, indices: "cudf.Series", separator: str = " "
    ) -> ParentType:
        """
        Combines tokens into strings by concatenating them in the order
        in which they appear in the ``indices`` column. The ``separator`` is
        concatenated between each token.

        Parameters
        ----------
        indices : Series
            Each value identifies the output row for the corresponding token.
        separator : str
            The string concatenated between each token in an output row.
            Default is space.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> strs = cudf.Series(["hello", "world", "one", "two", "three"])
        >>> indices = cudf.Series([0, 0, 1, 1, 2])
        >>> strs.str.detokenize(indices)
        0    hello world
        1        one two
        2          three
        dtype: object
        """
        separator = _massage_string_arg(separator, "separator")
        return self._return_or_inplace(
            cpp_detokenize(self._column, indices._column, separator),
            retain_index=False,
        )

    def character_tokenize(self) -> ParentType:
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
        if isinstance(self._parent, cudf.Series):
            return cudf.Series(result_col, name=self._parent.name)
        elif isinstance(self._parent, cudf.Index):
            return cudf.core.index.as_index(result_col, name=self._parent.name)
        else:
            return result_col

    def token_count(self, delimiter: str = " ") -> ParentType:
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
        if isinstance(delimiter, Column):
            return self._return_or_inplace(
                cpp_count_tokens_column(self._column, delimiter)
            )

        elif isinstance(delimiter, cudf.Scalar):
            return self._return_or_inplace(
                cpp_count_tokens_scalar(self._column, delimiter)
            )
        else:
            raise TypeError(
                f"Expected a Scalar or Column\
                for delimiters, but got {type(delimiter)}"
            )

    def ngrams(self, n: int = 2, separator: str = "_") -> ParentType:
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
        return self._return_or_inplace(
            cpp_generate_ngrams(self._column, n, separator), retain_index=False
        )

    def character_ngrams(self, n: int = 2) -> ParentType:
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
        return self._return_or_inplace(
            cpp_generate_character_ngrams(self._column, n), retain_index=False
        )

    def ngrams_tokenize(
        self, n: int = 2, delimiter: str = " ", separator: str = "_"
    ) -> ParentType:
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
        return self._return_or_inplace(
            cpp_ngrams_tokenize(self._column, n, delimiter, separator),
            retain_index=False,
        )

    def replace_tokens(
        self, targets, replacements, delimiter: str = None
    ) -> ParentType:
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
                cudf.Scalar(delimiter, dtype="str"),
            ),
        )

    def filter_tokens(
        self,
        min_token_length: int,
        replacement: str = None,
        delimiter: str = None,
    ) -> ParentType:
        """
        Remove tokens from within each string in the series that are
        smaller than min_token_length and optionally replace them
        with the replacement string.
        Tokens are identified by the delimiter character provided.

        Parameters
        ----------
        min_token_length: int
            Minimum number of characters for a token to be retained
            in the output string.

        replacement : str
            String used in place of removed tokens.

        delimiter : str
            The character(s) used to locate the tokens of each string.
            Default is whitespace.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["this is me", "theme music", ""])
        >>> sr.str.filter_tokens(3, replacement="_")
        0       this _ _
        1    theme music
        2
        dtype: object
        >>> sr = cudf.Series(["this;is;me", "theme;music", ""])
        >>> sr.str.filter_tokens(5,None,";")
        0             ;;
        1    theme;music
        2
        dtype: object
        """

        if replacement is None:
            replacement = ""
        elif not is_scalar(replacement):
            raise TypeError(
                f"Type of replacement should be a string,"
                f" found {type(replacement)}"
            )

        if delimiter is None:
            delimiter = ""
        elif not is_scalar(delimiter):
            raise TypeError(
                f"Type of delimiter should be a string,"
                f" found {type(delimiter)}"
            )

        return self._return_or_inplace(
            cpp_filter_tokens(
                self._column,
                min_token_length,
                cudf.Scalar(replacement, dtype="str"),
                cudf.Scalar(delimiter, dtype="str"),
            ),
        )

    def subword_tokenize(
        self,
        hash_file: str,
        max_length: int = 64,
        stride: int = 48,
        do_lower: bool = True,
        do_truncate: bool = False,
        max_rows_tensor: int = 500,
    ) -> Tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]:
        """
        Run CUDA BERT subword tokenizer on cuDF strings column.
        Encodes words to token ids using vocabulary from a pretrained
        tokenizer.

        This function requires about 21x the number of character bytes
        in the input strings column as working memory.

        Parameters
        ----------
        hash_file : str
            Path to hash file containing vocabulary of words with token-ids.
            This can be created from the raw vocabulary
            using the ``cudf.utils.hash_vocab_utils.hash_vocab`` function
        max_length : int, Default is 64
            Limits the length of the sequence returned.
            If tokenized string is shorter than max_length,
            output will be padded with 0s.
            If the tokenized string is longer than max_length and
            do_truncate == False, there will be multiple returned
            sequences containing the overflowing token-ids.
        stride : int, Default is 48
            If do_truncate == False and the tokenized string is larger
            than max_length, the sequences containing the overflowing
            token-ids can contain duplicated token-ids from the main
            sequence. If max_length is equal to stride there are no
            duplicated-id tokens. If stride is 80% of max_length,
            20% of the first sequence will be repeated on the second
            sequence and so on until the entire sentence is encoded.
        do_lower : bool, Default is True
            If set to true, original text will be lowercased before encoding.
        do_truncate : bool, Default is False
            If set to true, strings will be truncated and padded to
            max_length. Each input string will result in exactly one output
            sequence. If set to false, there may be multiple output
            sequences when the max_length is smaller than generated tokens.
        max_rows_tensor : int, Default is 500
            Maximum number of rows for the output token-ids expected
            to be generated by the tokenizer.
            Used for allocating temporary working memory on the GPU device.
            If the output generates a larger number of rows, behavior
            is undefined.
            This will vary based on stride, truncation, and max_length.
            For example, for non-overlapping sequences output rows
            will be the same as input rows.

        Returns
        -------
        token-ids : cupy.ndarray
            The token-ids for each string padded with 0s to max_length.
        attention-mask : cupy.ndarray
            The mask for token-ids result where corresponding positions
            identify valid token-id values.
        metadata : cupy.ndarray
            Each row contains the index id of the original string and the
            first and last index of the token-ids that are non-padded and
            non-overlapping.

        Examples
        --------
        >>> import cudf
        >>> from cudf.utils.hash_vocab_utils  import hash_vocab
        >>> hash_vocab('bert-base-uncased-vocab.txt', 'voc_hash.txt')
        >>> ser = cudf.Series(['this is the', 'best book'])
        >>> stride, max_length = 8, 8
        >>> max_rows_tensor = len(ser)
        >>> tokens, masks, metadata = ser.str.subword_tokenize('voc_hash.txt',
        ... max_length=max_length, stride=stride,
        ... max_rows_tensor=max_rows_tensor)
        >>> tokens.reshape(-1, max_length)
        array([[2023, 2003, 1996,    0,    0,    0,    0,    0],
               [2190, 2338,    0,    0,    0,    0,    0,    0]], dtype=uint32)
        >>> masks.reshape(-1, max_length)
        array([[1, 1, 1, 0, 0, 0, 0, 0],
               [1, 1, 0, 0, 0, 0, 0, 0]], dtype=uint32)
        >>> metadata.reshape(-1, 3)
        array([[0, 0, 2],
               [1, 0, 1]], dtype=uint32)
        """
        tokens, masks, metadata = cpp_subword_tokenize(
            self._column,
            hash_file,
            max_length,
            stride,
            do_lower,
            do_truncate,
            max_rows_tensor,
        )
        return (
            cupy.asarray(tokens),
            cupy.asarray(masks),
            cupy.asarray(metadata),
        )

    def porter_stemmer_measure(self) -> ParentType:
        """
        Compute the Porter Stemmer measure for each string.
        The Porter Stemmer algorithm is described `here
        <https://tartarus.org/martin/PorterStemmer/def.txt>`_.

        Returns
        -------
        Series or Index of object.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["hello", "super"])
        >>> ser.str.porter_stemmer_measure()
        0    1
        1    2
        dtype: int32
        """
        return self._return_or_inplace(
            cpp_porter_stemmer_measure(self._column)
        )

    def is_consonant(self, position) -> ParentType:
        """
        Return true for strings where the character at ``position`` is a
        consonant. The ``position`` parameter may also be a list of integers
        to check different characters per string.
        If the ``position`` is larger than the string length, False is
        returned for that string.

        Parameters
        ----------
        position: int or list-like
           The character position to check within each string.

        Returns
        -------
        Series or Index of bool dtype.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["toy", "trouble"])
        >>> ser.str.is_consonant(1)
        0    False
        1     True
        dtype: bool
        >>> positions = cudf.Series([2, 3])
        >>> ser.str.is_consonant(positions)
        0     True
        1    False
        dtype: bool
         """
        ltype = LetterType.CONSONANT

        if can_convert_to_column(position):
            return self._return_or_inplace(
                cpp_is_letter_multi(
                    self._column, ltype, column.as_column(position)
                ),
            )

        return self._return_or_inplace(
            cpp_is_letter(self._column, ltype, position)
        )

    def is_vowel(self, position) -> ParentType:
        """
        Return true for strings where the character at ``position`` is a
        vowel -- not a consonant. The ``position`` parameter may also be
        a list of integers to check different characters per string.
        If the ``position`` is larger than the string length, False is
        returned for that string.

        Parameters
        ----------
        position: int or list-like
           The character position to check within each string.

        Returns
        -------
        Series or Index of bool dtype.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series(["toy", "trouble"])
        >>> ser.str.is_vowel(1)
        0     True
        1    False
        dtype: bool
        >>> positions = cudf.Series([2, 3])
        >>> ser.str.is_vowel(positions)
        0    False
        1     True
        dtype: bool
        """
        ltype = LetterType.VOWEL

        if can_convert_to_column(position):
            return self._return_or_inplace(
                cpp_is_letter_multi(
                    self._column, ltype, column.as_column(position)
                ),
            )

        return self._return_or_inplace(
            cpp_is_letter(self._column, ltype, position)
        )

    def edit_distance(self, targets) -> ParentType:
        """
        The ``targets`` strings are measured against the strings in this
        instance using the Levenshtein edit distance algorithm.
        https://www.cuelogic.com/blog/the-levenshtein-algorithm

        The ``targets`` parameter may also be a single string in which
        case the edit distance is computed for all the strings against
        that single string.

        Parameters
        ----------
        targets : array-like, Sequence or Series or str
            The string(s) to measure against each string.

        Returns
        -------
        Series or Index of int32.

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["puppy", "doggy", "kitty"])
        >>> targets = cudf.Series(["pup", "dogie", "kitten"])
        >>> sr.str.edit_distance(targets=targets)
        0    2
        1    2
        2    2
        dtype: int32
        >>> sr.str.edit_distance("puppy")
        0    0
        1    4
        2    4
        dtype: int32
        """
        if is_scalar(targets):
            targets_column = column.as_column([targets])
        elif can_convert_to_column(targets):
            targets_column = column.as_column(targets)
        else:
            raise TypeError(
                f"targets should be an str, array-like or Series object, "
                f"found {type(targets)}"
            )

        return self._return_or_inplace(
            cpp_edit_distance(self._column, targets_column)
        )


def _massage_string_arg(value, name, allow_col=False):
    if isinstance(value, str):
        return cudf.Scalar(value, dtype="str")

    allowed_types = ["Scalar"]

    if allow_col:
        if isinstance(value, list):
            return column.as_column(value, dtype="str")

        if isinstance(value, Column) and is_string_dtype(value.dtype):
            return value

        allowed_types.append("Column")

    raise ValueError(
        f"Expected {_expected_types_format(allowed_types)} "
        f"for {name} but got {type(value)}"
    )


def _expected_types_format(types):
    if len(types) == 1:
        return types[0]

    return ", ".join(types[:-1]) + ", or " + types[-1]


class StringColumn(column.ColumnBase):
    """Implements operations for Columns of String type
    """

    _start_offset: Optional[int]
    _end_offset: Optional[int]
    _cached_sizeof: Optional[int]

    def __init__(
        self,
        mask: Buffer = None,
        size: int = None,  # TODO: make non-optional
        offset: int = 0,
        null_count: int = None,
        children: Tuple["column.ColumnBase", ...] = (),
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

        if len(children) == 0 and size != 0:
            # all nulls-column:
            offsets = column.full(size + 1, 0, dtype="int32")

            chars = cudf.core.column.as_column([], dtype="int8")
            children = (offsets, chars)

        super().__init__(
            data=None,
            size=size,
            dtype=dtype,
            mask=mask,
            offset=offset,
            null_count=null_count,
            children=children,
        )

        self._start_offset = None
        self._end_offset = None

    @property
    def start_offset(self) -> int:
        if self._start_offset is None:
            if (
                len(self.base_children) == 2
                and self.offset < self.base_children[0].size
            ):
                self._start_offset = int(
                    self.base_children[0].element_indexing(self.offset)
                )
            else:
                self._start_offset = 0

        return self._start_offset

    @property
    def end_offset(self) -> int:
        if self._end_offset is None:
            if (
                len(self.base_children) == 2
                and (self.offset + self.size) < self.base_children[0].size
            ):
                self._end_offset = int(
                    self.base_children[0].element_indexing(
                        self.offset + self.size
                    )
                )
            else:
                self._end_offset = 0

        return self._end_offset

    def __sizeof__(self) -> int:
        if self._cached_sizeof is None:
            n = 0
            if len(self.base_children) == 2:
                child0_size = (self.size + 1) * self.base_children[
                    0
                ].dtype.itemsize

                child1_size = (
                    self.end_offset - self.start_offset
                ) * self.base_children[1].dtype.itemsize

                n += child0_size + child1_size
            if self.nullable:
                n += cudf._lib.null_mask.bitmask_allocation_size_bytes(
                    self.size
                )
            self._cached_sizeof = n

        return self._cached_sizeof

    @property
    def base_size(self) -> int:
        if len(self.base_children) == 0:
            return 0
        else:
            return int(
                (self.base_children[0].size - 1)
                / self.base_children[0].dtype.itemsize
            )

    @property
    def data_array_view(self) -> cuda.devicearray.DeviceNDArray:
        raise ValueError("Cannot get an array view of a StringColumn")

    def sum(
        self, skipna: bool = None, dtype: Dtype = None, min_count: int = 0
    ):
        result_col = self._process_for_reduction(
            skipna=skipna, min_count=min_count
        )
        if isinstance(result_col, cudf.core.column.ColumnBase):
            return result_col.str().cat()
        else:
            return result_col

    def set_base_data(self, value):
        if value is not None:
            raise RuntimeError(
                "StringColumns do not use data attribute of Column, use "
                "`set_base_children` instead"
            )
        else:
            super().set_base_data(value)

    def set_base_mask(self, value: Optional[Buffer]):
        super().set_base_mask(value)

    def set_base_children(self, value: Tuple["column.ColumnBase", ...]):
        # TODO: Implement dtype validation of the children here somehow
        super().set_base_children(value)

    def __contains__(self, item: ScalarLike) -> bool:
        return True in self.str().contains(f"^{item}$")

    def str(self, parent: ParentType = None) -> StringMethods:
        return StringMethods(self, parent=parent)

    def unary_operator(self, unaryop: builtins.str):
        raise TypeError(
            f"Series of dtype `str` cannot perform the operation: "
            f"{unaryop}"
        )

    def __len__(self) -> int:
        return self.size

    @property
    def _nbytes(self) -> int:
        if self.size == 0:
            return 0
        else:
            return self.children[1].size

    def as_numerical_column(
        self, dtype: Dtype
    ) -> "cudf.core.column.NumericalColumn":
        out_dtype = np.dtype(dtype)

        if out_dtype.kind in {"i", "u"}:
            if not cpp_is_integer(self).all():
                raise ValueError(
                    "Could not convert strings to integer "
                    "type due to presence of non-integer values."
                )
        elif out_dtype.kind == "f":
            if not cpp_is_float(self).all():
                raise ValueError(
                    "Could not convert strings to float "
                    "type due to presence of non-floating values."
                )

        result_col = _str_to_numeric_typecast_functions[out_dtype](self)
        return result_col

    def _as_datetime_or_timedelta_column(self, dtype, format):
        if len(self) == 0:
            return cudf.core.column.as_column([], dtype=dtype)

        # Check for None strings
        if (self == "None").any():
            raise ValueError("Could not convert `None` value to datetime")

        casting_func = (
            str_cast.timestamp2int
            if dtype.type == np.datetime64
            else str_cast.timedelta2int
        )
        result_col = casting_func(self, dtype, format)

        boolean_match = self == "NaT"
        if (boolean_match).any():
            result_col[boolean_match] = None

        return result_col

    def as_datetime_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.DatetimeColumn":
        out_dtype = np.dtype(dtype)

        # infer on host from the first not na element
        # or return all null column if all values
        # are null in current column
        format = kwargs.get("format", None)
        if format is None:
            if self.null_count == len(self):
                return cast(
                    "cudf.core.column.DatetimeColumn",
                    column.column_empty(
                        len(self), dtype=out_dtype, masked=True
                    ),
                )
            else:
                format = datetime.infer_format(
                    self.apply_boolean_mask(self.notna()).element_indexing(0)
                )

        return self._as_datetime_or_timedelta_column(out_dtype, format)

    def as_timedelta_column(
        self, dtype: Dtype, **kwargs
    ) -> "cudf.core.column.TimeDeltaColumn":
        out_dtype = np.dtype(dtype)
        format = "%D days %H:%M:%S"
        return self._as_datetime_or_timedelta_column(out_dtype, format)

    def as_string_column(self, dtype: Dtype, format=None) -> StringColumn:
        return self

    @property
    def values_host(self) -> np.ndarray:
        """
        Return a numpy representation of the StringColumn.
        """
        return self.to_pandas().values

    @property
    def values(self) -> cupy.ndarray:
        """
        Return a CuPy representation of the StringColumn.
        """
        raise NotImplementedError(
            "String Arrays is not yet implemented in cudf"
        )

    def to_array(self, fillna: bool = None) -> np.ndarray:
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
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, Conversion to GPU array in strings is not yet "
            "supported.\nTo explicitly construct a host array, "
            "consider using .to_array()"
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    def serialize(self) -> Tuple[dict, list]:
        header = {"null_count": self.null_count}  # type: Dict[Any, Any]
        header["type-serialized"] = pickle.dumps(type(self))
        header["size"] = self.size

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
    def deserialize(cls, header: dict, frames: list) -> StringColumn:
        size = header["size"]
        if not isinstance(size, int):
            size = pickle.loads(size)

        # Deserialize the mask, value, and offset frames
        buffers = [Buffer(each_frame) for each_frame in frames]

        nbuf = None
        if header["null_count"] > 0:
            nbuf = buffers[2]

        children = []
        for h, b in zip(header["subheaders"], buffers[:2]):
            column_type = pickle.loads(h["type-serialized"])
            children.append(column_type.deserialize(h, [b]))

        col = cast(
            StringColumn,
            column.build_column(
                data=None,
                dtype="str",
                mask=nbuf,
                children=tuple(children),
                size=size,
            ),
        )
        return col

    def can_cast_safely(self, to_dtype: Dtype) -> bool:
        to_dtype = np.dtype(to_dtype)

        if self.dtype == to_dtype:
            return True
        elif to_dtype.kind in {"i", "u"} and not cpp_is_integer(self).all():
            return False
        elif to_dtype.kind == "f" and not cpp_is_float(self).all():
            return False
        else:
            return True

    def find_and_replace(
        self,
        to_replace: ColumnLike,
        replacement: ColumnLike,
        all_nan: bool = False,
    ) -> StringColumn:
        """
        Return col with *to_replace* replaced with *value*
        """

        to_replace_col = column.as_column(to_replace)
        if to_replace_col.null_count == len(to_replace_col):
            # If all of `to_replace` are `None`, dtype of `to_replace_col`
            # is inferred as `float64`, but this is a valid
            # string column too, Hence we will need to type-cast
            # to self.dtype.
            to_replace_col = to_replace_col.astype(self.dtype)

        replacement_col = column.as_column(replacement)
        if replacement_col.null_count == len(replacement_col):
            # If all of `replacement` are `None`, dtype of `replacement_col`
            # is inferred as `float64`, but this is a valid
            # string column too, Hence we will need to type-cast
            # to self.dtype.
            replacement_col = replacement_col.astype(self.dtype)

        if type(to_replace_col) != type(replacement_col):
            raise TypeError(
                f"to_replace and value should be of same types,"
                f"got to_replace dtype: {to_replace_col.dtype} and "
                f"value dtype: {replacement_col.dtype}"
            )

        if (
            to_replace_col.dtype != self.dtype
            and replacement_col.dtype != self.dtype
        ):
            return self.copy()

        return libcudf.replace.replace(self, to_replace_col, replacement_col)

    def fillna(
        self,
        fill_value: Any = None,
        method: builtins.str = None,
        dtype: Dtype = None,
    ) -> StringColumn:
        if fill_value is not None:
            if not is_scalar(fill_value):
                fill_value = column.as_column(fill_value, dtype=self.dtype)
            return super().fillna(value=fill_value, dtype="object")
        else:
            return super().fillna(method=method)

    def _find_first_and_last(self, value: ScalarLike) -> Tuple[int, int]:
        found_indices = self.str().contains(f"^{value}$")
        found_indices = libcudf.unary.cast(found_indices, dtype=np.int32)
        first = column.as_column(found_indices).find_first_value(1)
        last = column.as_column(found_indices).find_last_value(1)
        return first, last

    def find_first_value(
        self, value: ScalarLike, closest: bool = False
    ) -> int:
        return self._find_first_and_last(value)[0]

    def find_last_value(self, value: ScalarLike, closest: bool = False) -> int:
        return self._find_first_and_last(value)[1]

    def normalize_binop_value(self, other) -> "column.ColumnBase":
        # fastpath: gpu scalar
        if isinstance(other, cudf.Scalar) and other.dtype == "object":
            return column.as_column(other, length=len(self))
        if isinstance(other, column.ColumnBase):
            return other.astype(self.dtype)
        elif isinstance(other, str) or other is None:
            col = utils.scalar_broadcast_to(
                other, size=len(self), dtype="object"
            )
            return col
        elif isinstance(other, np.ndarray) and other.ndim == 0:
            col = utils.scalar_broadcast_to(
                other.item(), size=len(self), dtype="object"
            )
            return col
        else:
            raise TypeError(f"cannot broadcast {type(other)}")

    def default_na_value(self) -> ScalarLike:
        return None

    def binary_operator(
        self, op: builtins.str, rhs, reflect: bool = False
    ) -> "column.ColumnBase":
        lhs = self
        if reflect:
            lhs, rhs = rhs, lhs
        if isinstance(rhs, (StringColumn, str, cudf.Scalar)):
            if op == "add":
                return cast("column.ColumnBase", lhs.str().cat(others=rhs))
            elif op in ("eq", "ne", "gt", "lt", "ge", "le"):
                return _string_column_binop(self, rhs, op=op, out_dtype="bool")

        raise TypeError(
            f"{op} operator not supported between {type(self)} and {type(rhs)}"
        )

    @property
    def is_unique(self) -> bool:
        return len(self.unique()) == len(self)

    @property
    def __cuda_array_interface__(self):
        raise NotImplementedError(
            "Strings are not yet supported via `__cuda_array_interface__`"
        )

    @copy_docstring(column.ColumnBase.view)
    def view(self, dtype) -> "cudf.core.column.ColumnBase":
        if self.null_count > 0:
            raise ValueError(
                "Can not produce a view of a string column with nulls"
            )
        dtype = np.dtype(dtype)
        str_byte_offset = self.base_children[0].element_indexing(self.offset)
        str_end_byte_offset = self.base_children[0].element_indexing(
            self.offset + self.size
        )
        char_dtype_size = self.base_children[1].dtype.itemsize

        n_bytes_to_view = (
            str_end_byte_offset - str_byte_offset
        ) * char_dtype_size

        to_view = column.build_column(
            self.base_children[1].data,
            dtype=self.base_children[1].dtype,
            offset=str_byte_offset,
            size=n_bytes_to_view,
        )

        return to_view.view(dtype)


@annotate("BINARY_OP", color="orange", domain="cudf_python")
def _string_column_binop(
    lhs: "column.ColumnBase",
    rhs: "column.ColumnBase",
    op: str,
    out_dtype: Dtype,
) -> "column.ColumnBase":
    out = libcudf.binaryop.binaryop(lhs=lhs, rhs=rhs, op=op, dtype=out_dtype)
    return out


def _get_cols_list(parent_obj, others):

    parent_index = (
        parent_obj.index if isinstance(parent_obj, cudf.Series) else parent_obj
    )

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
        cols_list = [
            column.as_column(frame.reindex(parent_index), dtype="str")
            if (
                parent_index is not None
                and isinstance(frame, cudf.Series)
                and not frame.index.equals(parent_index)
            )
            else column.as_column(frame, dtype="str")
            for frame in others
        ]

        return cols_list
    elif others is not None:
        if (
            parent_index is not None
            and isinstance(others, cudf.Series)
            and not others.index.equals(parent_index)
        ):
            others = others.reindex(parent_index)

        return [column.as_column(others, dtype="str")]
    else:
        raise TypeError(
            "others must be Series, Index, DataFrame, np.ndarrary "
            "or list-like (either containing only strings or "
            "containing only objects of type Series/Index/"
            "np.ndarray[1-dim])"
        )
