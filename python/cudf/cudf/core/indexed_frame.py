# Copyright (c) 2021-2024, NVIDIA CORPORATION.
"""Base class for Frame types that have an index."""

from __future__ import annotations

import numbers
import operator
import textwrap
import warnings
from collections import Counter, abc
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from uuid import uuid4

import cupy as cp
import numpy as np
import pandas as pd
from typing_extensions import Self

import cudf
import cudf._lib as libcudf
from cudf._typing import (
    ColumnLike,
    DataFrameOrSeries,
    Dtype,
    NotImplementedType,
)
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    is_bool_dtype,
    is_decimal_dtype,
    is_dict_like,
    is_list_like,
    is_scalar,
)
from cudf.core._base_index import BaseIndex
from cudf.core._compat import PANDAS_LT_300
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import ColumnBase, as_column
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import BooleanMask, GatherMap
from cudf.core.dtypes import ListDtype
from cudf.core.frame import Frame
from cudf.core.groupby.groupby import GroupBy
from cudf.core.index import Index, RangeIndex, _index_from_data
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import _Resampler
from cudf.core.udf.utils import (
    _compile_or_get,
    _get_input_args_from_frame,
    _post_process_output_col,
    _return_arr_from_dtype,
)
from cudf.core.window import Rolling
from cudf.utils import docutils, ioutils
from cudf.utils._numba import _CUDFNumbaConfig
from cudf.utils.docutils import copy_docstring
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import _warn_no_dask_cudf

doc_reset_index_template = """
        Reset the index of the {klass}, or a level of it.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
{argument}
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        {return_type}
            {klass} with the new index or None if ``inplace=True``.{return_doc}

        Examples
        --------
        {example}
"""


doc_binop_template = textwrap.dedent(
    """
    Get {operation} of DataFrame or Series and other, element-wise (binary
    operator `{op_name}`).

    Equivalent to ``frame + other``, but with support to substitute a
    ``fill_value`` for missing data in one of the inputs.

    Parameters
    ----------
    other : scalar, sequence, Series, or DataFrame
        Any single or multiple element data structure, or list-like object.
    axis : int or string
        Only ``0`` is supported for series, ``1`` or ``columns`` supported
        for dataframe
    level : int or name
        Broadcast across a level, matching Index values on the
        passed MultiIndex level. Not yet supported.
    fill_value  : float or None, default None
        Fill existing missing (NaN) values, and any new element needed
        for successful DataFrame alignment, with this value before
        computation. If data in both corresponding DataFrame locations
        is missing the result will be missing.

    Returns
    -------
    DataFrame or Series
        Result of the arithmetic operation.

    Examples
    --------

    **DataFrame**

    >>> df = cudf.DataFrame(
    ...     {{'angles': [0, 3, 4], 'degrees': [360, 180, 360]}},
    ...     index=['circle', 'triangle', 'rectangle']
    ... )
    {df_op_example}

    **Series**

    >>> a = cudf.Series([1, 1, 1, None], index=['a', 'b', 'c', 'd'])
    >>> b = cudf.Series([1, None, 1, None], index=['a', 'b', 'd', 'e'])
    {ser_op_example}
    """
)


def _get_host_unique(array):
    if isinstance(array, (cudf.Series, cudf.Index, ColumnBase)):
        return array.unique.to_pandas()
    elif isinstance(array, (str, numbers.Number)):
        return [array]
    else:
        return set(array)


def _drop_columns(f: Frame, columns: abc.Iterable, errors: str):
    for c in columns:
        try:
            f._drop_column(c)
        except KeyError as e:
            if errors == "ignore":
                pass
            else:
                raise e


def _indices_from_labels(obj, labels):
    if not isinstance(labels, cudf.MultiIndex):
        labels = cudf.core.column.as_column(labels)

        if isinstance(obj.index.dtype, cudf.CategoricalDtype):
            labels = labels.astype("category")
            codes = labels.codes.astype(obj.index.codes.dtype)
            labels = cudf.core.column.build_categorical_column(
                categories=labels.dtype.categories,
                codes=codes,
                ordered=labels.dtype.ordered,
            )
        else:
            labels = labels.astype(obj.index.dtype)

    # join is not guaranteed to maintain the index ordering
    # so we will sort it with its initial ordering which is stored
    # in column "__"
    lhs = cudf.DataFrame({"__": as_column(range(len(labels)))}, index=labels)
    rhs = cudf.DataFrame({"_": as_column(range(len(obj)))}, index=obj.index)
    return lhs.join(rhs).sort_values(by=["__", "_"])["_"]


def _get_label_range_or_mask(index, start, stop, step):
    if (
        not (start is None and stop is None)
        and type(index) is cudf.core.index.DatetimeIndex
        and index.is_monotonic_increasing is False
    ):
        start = pd.to_datetime(start)
        stop = pd.to_datetime(stop)
        if start is not None and stop is not None:
            if start > stop:
                return slice(0, 0, None)
            if (start in index) and (stop in index):
                # when we have a non-monotonic datetime index, return
                # values in the slice defined by index_of(start) and
                # index_of(end)
                start_loc = index.get_loc(start.to_datetime64())
                stop_loc = index.get_loc(stop.to_datetime64()) + 1
                return slice(start_loc, stop_loc)
            else:
                raise KeyError(
                    "Value based partial slicing on non-monotonic "
                    "DatetimeIndexes with non-existing keys is not allowed.",
                )
        elif start is not None:
            boolean_mask = index >= start
        else:
            boolean_mask = index <= stop
        return boolean_mask
    else:
        return index.find_label_range(slice(start, stop, step))


class _FrameIndexer:
    """Parent class for indexers."""

    def __init__(self, frame):
        self._frame = frame


_LocIndexerClass = TypeVar("_LocIndexerClass", bound="_FrameIndexer")
_IlocIndexerClass = TypeVar("_IlocIndexerClass", bound="_FrameIndexer")


class IndexedFrame(Frame):
    """A frame containing an index.

    This class encodes the common behaviors for core user-facing classes like
    DataFrame and Series that consist of a sequence of columns along with a
    special set of index columns.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    # mypy can't handle bound type variables as class members
    _loc_indexer_type: Type[_LocIndexerClass]  # type: ignore
    _iloc_indexer_type: Type[_IlocIndexerClass]  # type: ignore
    _index: cudf.core.index.BaseIndex
    _groupby = GroupBy
    _resampler = _Resampler

    _VALID_SCANS = {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
    }

    # Necessary because the function names don't directly map to the docs.
    _SCAN_DOCSTRINGS = {
        "cumsum": {"op_name": "cumulative sum"},
        "cumprod": {"op_name": "cumulative product"},
        "cummin": {"op_name": "cumulative min"},
        "cummax": {"op_name": "cumulative max"},
    }

    def __init__(self, data=None, index=None):
        super().__init__(data=data)
        # TODO: Right now it is possible to initialize an IndexedFrame without
        # an index. The code's correctness relies on the subclass constructors
        # assigning the attribute after the fact. We should restructure those
        # to ensure that this constructor is always invoked with an index.
        self._index = index

    @property
    def _num_rows(self) -> int:
        # Important to use the index because the data may be empty.
        return len(self._index)

    @property
    def _index_names(self) -> Tuple[Any, ...]:  # TODO: Tuple[str]?
        return self._index._data.names

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[BaseIndex] = None,
    ):
        out = super()._from_data(data)
        out._index = RangeIndex(out._data.nrows) if index is None else index
        return out

    @_cudf_nvtx_annotate
    def _from_data_like_self(self, data: MutableMapping):
        out = self._from_data(data, self._index)
        out._data._level_names = self._data._level_names
        return out

    @_cudf_nvtx_annotate
    def _from_columns_like_self(
        self,
        columns: List[ColumnBase],
        column_names: Optional[abc.Iterable[str]] = None,
        index_names: Optional[List[str]] = None,
        *,
        override_dtypes: Optional[abc.Iterable[Optional[Dtype]]] = None,
    ) -> Self:
        """Construct a `Frame` from a list of columns with metadata from self.

        If `index_names` is set, the first `len(index_names)` columns are
        used to construct the index of the frame.

        If override_dtypes is provided then any non-None entry will be
        used for the dtype of the matching column in preference to the
        dtype of the column in self.
        """
        if column_names is None:
            column_names = self._column_names

        data_columns = columns
        index = None

        if index_names is not None:
            n_index_columns = len(index_names)
            data_columns = columns[n_index_columns:]
            index = _index_from_data(
                dict(enumerate(columns[:n_index_columns]))
            )
            if isinstance(index, cudf.MultiIndex):
                index.names = index_names
            else:
                index.name = index_names[0]

        data = dict(zip(column_names, data_columns))
        frame = self.__class__._from_data(data)

        if index is not None:
            frame._index = index
        return frame._copy_type_metadata(
            self,
            include_index=bool(index_names),
            override_dtypes=override_dtypes,
        )

    def __round__(self, digits=0):
        # Shouldn't be added to BinaryOperand
        # because pandas Index doesn't implement
        # this method.
        return self.round(decimals=digits)

    def _mimic_inplace(
        self, result: Self, inplace: bool = False
    ) -> Optional[Self]:
        if inplace:
            self._index = result._index
        return super()._mimic_inplace(result, inplace)

    # Scans
    @_cudf_nvtx_annotate
    def _scan(self, op, axis=None, skipna=True):
        """
        Return {op_name} of the {cls}.

        Parameters
        ----------
        axis: {{index (0), columns(1)}}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        {cls}

        Examples
        --------
        **Series**

        >>> import cudf
        >>> ser = cudf.Series([1, 5, 2, 4, 3])
        >>> ser.cumsum()
        0    1
        1    6
        2    8
        3    12
        4    15

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({{'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]}})
        >>> s.cumsum()
            a   b
        0   1   7
        1   3  15
        2   6  24
        3  10  34
        """
        cast_to_int = op in ("cumsum", "cumprod")
        skipna = True if skipna is None else skipna

        results = {}
        for name, col in self._data.items():
            if skipna:
                try:
                    result_col = col.nans_to_nulls()
                except AttributeError:
                    result_col = col
            else:
                if col.has_nulls(include_nan=True):
                    first_index = col.isnull().find_first_value(True)
                    result_col = col.copy()
                    result_col[first_index:] = None
                else:
                    result_col = col

            if (
                cast_to_int
                and not is_decimal_dtype(result_col.dtype)
                and (
                    np.issubdtype(result_col.dtype, np.integer)
                    or np.issubdtype(result_col.dtype, np.bool_)
                )
            ):
                # For reductions that accumulate a value (e.g. sum, not max)
                # pandas returns an int64 dtype for all int or bool dtypes.
                result_col = result_col.astype(np.int64)
            results[name] = getattr(result_col, op)()
        return self._from_data(results, self._index)

    def _check_data_index_length_match(self) -> None:
        # Validate that the number of rows in the data matches the index if the
        # data is not empty. This is a helper for the constructor.
        if self._data.nrows > 0 and self._data.nrows != len(self._index):
            raise ValueError(
                f"Length of values ({self._data.nrows}) does not "
                f"match length of index ({len(self._index)})"
            )

    @property
    @_cudf_nvtx_annotate
    def empty(self):
        """
        Indicator whether DataFrame or Series is empty.

        True if DataFrame/Series is entirely empty (no items),
        meaning any of the axes are of length 0.

        Returns
        -------
        out : bool
            If DataFrame/Series is empty, return True, if not return False.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A' : []})
        >>> df
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df.empty
        True

        If we only have `null` values in our DataFrame, it is
        not considered empty! We will need to drop
        the `null`'s to make the DataFrame empty:

        >>> df = cudf.DataFrame({'A' : [None, None]})
        >>> df
              A
        0  <NA>
        1  <NA>
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        Non-empty and empty Series example:

        >>> s = cudf.Series([1, 2, None])
        >>> s
        0       1
        1       2
        2    <NA>
        dtype: int64
        >>> s.empty
        False
        >>> s = cudf.Series([])
        >>> s
        Series([], dtype: float64)
        >>> s.empty
        True

        .. pandas-compat::
            **DataFrame.empty, Series.empty**

            If DataFrame/Series contains only `null` values, it is still not
            considered empty. See the example above.
        """
        return self.size == 0

    @_cudf_nvtx_annotate
    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""

        return cudf.io.json.to_json(
            self, path_or_buf=path_or_buf, *args, **kwargs
        )

    @_cudf_nvtx_annotate
    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""

        cudf.io.hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @_cudf_nvtx_annotate
    def to_string(self):
        r"""
        Convert to string

        cuDF uses Pandas internals for efficient string formatting.
        Set formatting options using pandas string formatting options and
        cuDF objects will print identically to Pandas objects.

        cuDF supports `null/None` as a value in any column type, which
        is transparently supported during this output process.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2]
        >>> df['val'] = [float(i + 10) for i in range(3)]
        >>> df.to_string()
        '   key   val\n0    0  10.0\n1    1  11.0\n2    2  12.0'
        """
        return str(self)

    def copy(self, deep: bool = True) -> Self:
        """Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).
        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). Any changes to the data of the original
        will be reflected in the shallow copy (and vice versa).

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        copy : Series or DataFrame
            Object type matches caller.

        Examples
        --------
        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        dtype: int64

        **Shallow copy versus default (deep) copy:**

        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> deep = s.copy()
        >>> shallow = s.copy(deep=False)

        Updates to the data shared by shallow copy and original is reflected
        in both; deep copy remains unchanged.

        >>> s['a'] = 3
        >>> shallow['b'] = 4
        >>> s
        a    3
        b    4
        dtype: int64
        >>> shallow
        a    3
        b    4
        dtype: int64
        >>> deep
        a    1
        b    2
        dtype: int64
        """
        return self._from_data(
            self._data.copy(deep=deep),
            # Indexes are immutable so copies can always be shallow.
            self._index.copy(deep=False),
        )

    @_cudf_nvtx_annotate
    def equals(self, other):  # noqa: D102
        if not super().equals(other):
            return False
        return self._index.equals(other._index)

    @property
    def index(self):
        """Get the labels for the rows."""
        return self._index

    @index.setter
    def index(self, value):
        old_length = len(self)
        new_length = len(value)

        # A DataFrame with 0 columns can have an index of arbitrary length.
        if len(self._data) > 0 and new_length != old_length:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_length} elements, "
                f"new values have {len(value)} elements"
            )
        self._index = Index(value)

    @_cudf_nvtx_annotate
    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace=False,
        limit=None,
        regex=False,
        method=no_default,
    ):
        """Replace values given in ``to_replace`` with ``value``.

        Parameters
        ----------
        to_replace : numeric, str or list-like
            Value(s) to replace.

            * numeric or str:
                - values equal to ``to_replace`` will be replaced
                  with ``value``
            * list of numeric or str:
                - If ``value`` is also list-like, ``to_replace`` and
                  ``value`` must be of same length.
            * dict:
                - Dicts can be used to specify different replacement values
                  for different existing values. For example, {'a': 'b',
                  'y': 'z'} replaces the value 'a' with 'b' and
                  'y' with 'z'.
                  To use a dict in this way the ``value`` parameter should
                  be ``None``.
        value : scalar, dict, list-like, str, default None
            Value to replace any values matching ``to_replace`` with.
        inplace : bool, default False
            If True, in place.

        See Also
        --------
        Series.fillna

        Raises
        ------
        TypeError
            - If ``to_replace`` is not a scalar, array-like, dict, or None
            - If ``to_replace`` is a dict and value is not a list, dict,
              or Series
        ValueError
            - If a list is passed to ``to_replace`` and ``value`` but they
              are not the same length.

        Returns
        -------
        result : Series
            Series after replacement. The mask and index are preserved.

        Examples
        --------
        **Series**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> s = cudf.Series([0, 1, 2, 3, 4])
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        dtype: int64
        >>> s.replace(0, 5)
        0    5
        1    1
        2    2
        3    3
        4    4
        dtype: int64

        List-like ``to_replace``

        >>> s.replace([1, 2], 10)
        0     0
        1    10
        2    10
        3     3
        4     4
        dtype: int64

        dict-like ``to_replace``

        >>> s.replace({1:5, 3:50})
        0     0
        1     5
        2     2
        3    50
        4     4
        dtype: int64
        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0     b
        1     a
        2     a
        3     b
        4     a
        dtype: object
        >>> s.replace({'a': None})
        0       b
        1    <NA>
        2    <NA>
        3       b
        4    <NA>
        dtype: object

        If there is a mismatch in types of the values in
        ``to_replace`` & ``value`` with the actual series, then
        cudf exhibits different behavior with respect to pandas
        and the pairs are ignored silently:

        >>> s = cudf.Series(['b', 'a', 'a', 'b', 'a'])
        >>> s
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object
        >>> s.replace('a', 1)
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object
        >>> s.replace(['a', 'c'], [1, 2])
        0    b
        1    a
        2    a
        3    b
        4    a
        dtype: object

        **DataFrame**

        Scalar ``to_replace`` and ``value``

        >>> import cudf
        >>> df = cudf.DataFrame({'A': [0, 1, 2, 3, 4],
        ...                    'B': [5, 6, 7, 8, 9],
        ...                    'C': ['a', 'b', 'c', 'd', 'e']})
        >>> df
           A  B  C
        0  0  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e
        >>> df.replace(0, 5)
           A  B  C
        0  5  5  a
        1  1  6  b
        2  2  7  c
        3  3  8  d
        4  4  9  e

        List-like ``to_replace``

        >>> df.replace([0, 1, 2, 3], 4)
           A  B  C
        0  4  5  a
        1  4  6  b
        2  4  7  c
        3  4  8  d
        4  4  9  e
        >>> df.replace([0, 1, 2, 3], [4, 3, 2, 1])
           A  B  C
        0  4  5  a
        1  3  6  b
        2  2  7  c
        3  1  8  d
        4  4  9  e

        dict-like ``to_replace``

        >>> df.replace({0: 10, 1: 100})
             A  B  C
        0   10  5  a
        1  100  6  b
        2    2  7  c
        3    3  8  d
        4    4  9  e
        >>> df.replace({'A': 0, 'B': 5}, 100)
             A    B  C
        0  100  100  a
        1    1    6  b
        2    2    7  c
        3    3    8  d
        4    4    9  e

        .. pandas-compat::
            **DataFrame.replace, Series.replace**

            Parameters that are currently not supported are: `limit`, `regex`,
            `method`
        """
        if limit is not None:
            raise NotImplementedError("limit parameter is not implemented yet")

        if regex:
            raise NotImplementedError("regex parameter is not implemented yet")

        if method is not no_default:
            warnings.warn(
                "The 'method' keyword in "
                f"{type(self).__name__}.replace is deprecated and "
                "will be removed in a future version.",
                FutureWarning,
            )
        elif method not in {"pad", None, no_default}:
            raise NotImplementedError("method parameter is not implemented")

        if (
            value is no_default
            and method is no_default
            and not is_dict_like(to_replace)
            and regex is False
        ):
            warnings.warn(
                f"{type(self).__name__}.replace without 'value' and with "
                "non-dict-like 'to_replace' is deprecated "
                "and will raise in a future version. "
                "Explicitly specify the new values instead.",
                FutureWarning,
            )
        if not (to_replace is None and value is no_default):
            copy_data = {}
            (
                all_na_per_column,
                to_replace_per_column,
                replacements_per_column,
            ) = _get_replacement_values_for_columns(
                to_replace=to_replace,
                value=value,
                columns_dtype_map=self._dtypes,
            )

            for name, col in self._data.items():
                try:
                    copy_data[name] = col.find_and_replace(
                        to_replace_per_column[name],
                        replacements_per_column[name],
                        all_na_per_column[name],
                    )
                except (KeyError, OverflowError):
                    # We need to create a deep copy if:
                    # i. `find_and_replace` was not successful or any of
                    #    `to_replace_per_column`, `replacements_per_column`,
                    #    `all_na_per_column` don't contain the `name`
                    #    that exists in `copy_data`.
                    # ii. There is an OverflowError while trying to cast
                    #     `to_replace_per_column` to `replacements_per_column`.
                    copy_data[name] = col.copy(deep=True)
        else:
            copy_data = self._data.copy(deep=True)

        result = self._from_data(copy_data, self._index)

        return self._mimic_inplace(result, inplace=inplace)

    @_cudf_nvtx_annotate
    def clip(self, lower=None, upper=None, inplace=False, axis=1):
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.
        Thresholds can be singular values or array like,
        and in the latter case the clipping is performed
        element-wise in the specified axis. Currently only
        `axis=1` is supported.

        Parameters
        ----------
        lower : scalar or array_like, default None
            Minimum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on lower.
            In case of Series/Index, lower is expected to be
            a scalar or an array of size 1.
        upper : scalar or array_like, default None
            Maximum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on upper.
            In case of Series, upper is expected to be
            a scalar or an array of size 1.
        inplace : bool, default False

        Returns
        -------
        Clipped DataFrame/Series/Index/MultiIndex

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":['a', 'b', 'c', 'd']})
        >>> df.clip(lower=[2, 'b'], upper=[3, 'c'])
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=None, upper=[3, 'c'])
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=[2, 'b'], upper=None)
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  4  d

        >>> df.clip(lower=2, upper=3, inplace=True)
        >>> df
           a  b
        0  2  2
        1  2  3
        2  3  3
        3  3  3

        >>> import cudf
        >>> sr = cudf.Series([1, 2, 3, 4])
        >>> sr.clip(lower=2, upper=3)
        0    2
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=None, upper=3)
        0    1
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=2, upper=None, inplace=True)
        >>> sr
        0    2
        1    2
        2    3
        3    4
        dtype: int64
        """
        if axis != 1:
            raise NotImplementedError("`axis is not yet supported in clip`")

        if lower is None and upper is None:
            return None if inplace is True else self.copy(deep=True)

        if is_scalar(lower):
            lower = np.full(self._num_columns, lower)
        if is_scalar(upper):
            upper = np.full(self._num_columns, upper)

        if len(lower) != len(upper):
            raise ValueError("Length of lower and upper should be equal")

        if len(lower) != self._num_columns:
            raise ValueError(
                "Length of lower/upper should be equal to number of columns"
            )

        if self.ndim == 1:
            # In case of series and Index,
            # swap lower and upper if lower > upper
            if (
                lower[0] is not None
                and upper[0] is not None
                and (lower[0] > upper[0])
            ):
                lower[0], upper[0] = upper[0], lower[0]

        data = {
            name: col.clip(lower[i], upper[i])
            for i, (name, col) in enumerate(self._data.items())
        }
        output = self._from_data(data, self._index)
        output._copy_type_metadata(self, include_index=False)
        return self._mimic_inplace(output, inplace=inplace)

    @_cudf_nvtx_annotate
    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        This function only applies to elements that are all numeric.

        Returns
        -------
        DataFrame/Series
            Absolute value of each element.

        Examples
        --------
        Absolute numeric values in a Series

        >>> s = cudf.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        dtype: float64
        """
        return self._unaryop("abs")

    @_cudf_nvtx_annotate
    def dot(self, other, reflect=False):
        """
        Get dot product of frame and other, (binary operator `dot`).

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`,
        `dot`) to arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`,
        `@`.

        Parameters
        ----------
        other : Sequence, Series, or DataFrame
            Any multiple element data structure, or list-like object.
        reflect : bool, default False
            If ``True``, swap the order of the operands. See
            https://docs.python.org/3/reference/datamodel.html#object.__ror__
            for more information on when this is necessary.

        Returns
        -------
        scalar, Series, or DataFrame
            The result of the operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([[1, 2, 3, 4],
        ...                      [5, 6, 7, 8]])
        >>> df @ df.T
            0    1
        0  30   70
        1  70  174
        >>> s = cudf.Series([1, 1, 1, 1])
        >>> df @ s
        0    10
        1    26
        dtype: int64
        >>> [1, 2, 3, 4] @ s
        10
        """
        # TODO: This function does not currently support nulls.
        lhs = self.values
        result_index = None
        result_cols = None
        if isinstance(self, cudf.Series) and isinstance(
            other, (cudf.Series, cudf.DataFrame)
        ):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("matrices are not aligned")

            lhs = self.reindex(index=common, copy=False).values
            rhs = other.reindex(index=common, copy=False).values
            if isinstance(other, cudf.DataFrame):
                result_index = other._data.to_pandas_index()
        elif isinstance(self, cudf.DataFrame) and isinstance(
            other, (cudf.Series, cudf.DataFrame)
        ):
            common = self._data.to_pandas_index().union(
                other.index.to_pandas()
            )
            if len(common) > len(self._data.names) or len(common) > len(
                other.index
            ):
                raise ValueError("matrices are not aligned")

            lhs = self.reindex(columns=common, copy=False)
            result_index = lhs.index

            rhs = other.reindex(index=common, copy=False).values
            lhs = lhs.values
            if isinstance(other, cudf.DataFrame):
                result_cols = other._data.to_pandas_index()

        elif isinstance(
            other, (cp.ndarray, np.ndarray)
        ) or cudf.utils.dtypes.can_convert_to_column(other):
            rhs = cp.asarray(other)
        else:
            # TODO: This should raise an exception, not return NotImplemented,
            # but __matmul__ relies on the current behavior. We should either
            # move this implementation to __matmul__ and call it from here
            # (checking for NotImplemented and raising NotImplementedError if
            # that's what's returned), or __matmul__ should catch a
            # NotImplementedError from here and return NotImplemented. The
            # latter feels cleaner (putting the implementation in this method
            # rather than in the operator) but will be slower in the (highly
            # unlikely) case that we're multiplying a cudf object with another
            # type of object that somehow supports this behavior.
            return NotImplemented
        if reflect:
            lhs, rhs = rhs, lhs

        result = lhs.dot(rhs)
        if len(result.shape) == 1:
            return cudf.Series(
                result,
                index=self.index if result_index is None else result_index,
            )
        if len(result.shape) == 2:
            return cudf.DataFrame(
                result,
                index=self.index if result_index is None else result_index,
                columns=result_cols,
            )
        return result.item()

    @_cudf_nvtx_annotate
    def __matmul__(self, other):
        return self.dot(other)

    @_cudf_nvtx_annotate
    def __rmatmul__(self, other):
        return self.dot(other, reflect=True)

    @_cudf_nvtx_annotate
    def head(self, n=5):
        """
        Return the first `n` rows.
        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.
        For negative values of `n`, this function returns all rows except
        the last `n` rows, equivalent to ``df[:-n]``.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        DataFrame or Series
            The first `n` rows of the caller object.

        Examples
        --------
        **Series**

        >>> ser = cudf.Series(['alligator', 'bee', 'falcon',
        ... 'lion', 'monkey', 'parrot', 'shark', 'whale', 'zebra'])
        >>> ser
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        6        shark
        7        whale
        8        zebra
        dtype: object

        Viewing the first 5 lines

        >>> ser.head()
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        dtype: object

        Viewing the first `n` lines (three in this case)

        >>> ser.head(3)
        0    alligator
        1          bee
        2       falcon
        dtype: object

        For negative values of `n`

        >>> ser.head(-3)
        0    alligator
        1          bee
        2       falcon
        3         lion
        4       monkey
        5       parrot
        dtype: object

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.head(2)
           key   val
        0    0  10.0
        1    1  11.0
        """
        return self.iloc[:n]

    @_cudf_nvtx_annotate
    def tail(self, n=5):
        """
        Returns the last n rows as a new DataFrame or Series

        Examples
        --------
        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> df.tail(2)
           key   val
        3    3  13.0
        4    4  14.0

        **Series**

        >>> import cudf
        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.tail(2)
        3    1
        4    0
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

    @_cudf_nvtx_annotate
    def pipe(self, func, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``.

        Parameters
        ----------
        func : function
            Function to apply to the Series/DataFrame.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the Series/DataFrame.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Examples
        --------
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects. Instead of writing

        >>> func(g(h(df), arg1=a), arg2=b, arg3=c)

        You can write

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe(func, arg2=b, arg3=c)
        ... )

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe((func, 'arg2'), arg1=a, arg3=c)
        ...  )
        """
        return cudf.core.common.pipe(self, func, *args, **kwargs)

    @_cudf_nvtx_annotate
    def sum(
        self,
        axis=no_default,
        skipna=True,
        dtype=None,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ):
        """
        Return sum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.sum()
        a    10
        b    34
        dtype: int64

        .. pandas-compat::
            **DataFrame.sum, Series.sum**

            Parameters currently not supported are `level`, `numeric_only`.
        """
        return self._reduce(
            "sum",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def product(
        self,
        axis=no_default,
        skipna=True,
        dtype=None,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.product()
        a      24
        b    5040
        dtype: int64

        .. pandas-compat::
            **DataFrame.product, Series.product**

            Parameters currently not supported are level`, `numeric_only`.
        """

        return self._reduce(
            # cuDF columns use "product" as the op name, but cupy uses "prod"
            # and we need cupy if axis == 1.
            "prod" if axis in {1, "columns"} else "product",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    # Alias for pandas compatibility.
    prod = product

    @_cudf_nvtx_annotate
    def mean(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        """
        Return the mean of the values for the requested axis.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        mean : Series or DataFrame (if level specified)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.mean()
        a    2.5
        b    8.5
        dtype: float64
        """
        return self._reduce(
            "mean",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    def median(
        self, axis=None, skipna=True, level=None, numeric_only=None, **kwargs
    ):
        """
        Return the median of the values for the requested axis.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on. For Series this
            parameter is unused and defaults to 0.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        scalar

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 3, 25, 24, 6])
        >>> ser
        0    10
        1    25
        2     3
        3    25
        4    24
        5     6
        dtype: int64
        >>> ser.median()
        17.0

        .. pandas-compat::
            **DataFrame.median, Series.median**

            Parameters currently not supported are `level` and `numeric_only`.
        """
        return self._reduce(
            "median",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def std(
        self,
        axis=no_default,
        skipna=True,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        """
        Return sample standard deviation of the DataFrame.

        Normalized by N-1 by default. This can be changed using
        the `ddof` argument

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations
            is N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.std()
        a    1.290994
        b    1.290994
        dtype: float64

        .. pandas-compat::
            **DataFrame.std, Series.std**

            Parameters currently not supported are `level` and
            `numeric_only`
        """

        return self._reduce(
            "std",
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def var(
        self,
        axis=no_default,
        skipna=True,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        """
        Return unbiased variance of the DataFrame.

        Normalized by N-1 by default. This can be changed using the
        ddof argument.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is
            N - ddof, where N represents the number of elements.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        scalar

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.var()
        a    1.666667
        b    1.666667
        dtype: float64

        .. pandas-compat::
            **DataFrame.var, Series.var**

            Parameters currently not supported are `level` and
            `numeric_only`
        """
        return self._reduce(
            "var",
            axis=axis,
            skipna=skipna,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def kurtosis(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher's definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series or scalar

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4])
        >>> series.kurtosis()
        -1.1999999999999904

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.kurt()
        a   -1.2
        b   -1.2
        dtype: float64

        .. pandas-compat::
            **DataFrame.kurtosis**

            Parameters currently not supported are `level` and `numeric_only`
        """
        if axis not in (0, "index", None, no_default):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    # Alias for kurtosis.
    kurt = kurtosis

    @_cudf_nvtx_annotate
    def skew(self, axis=0, skipna=True, numeric_only=False, **kwargs):
        """
        Return unbiased Fisher-Pearson skew of a sample.

        Parameters
        ----------
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        numeric_only : bool, default False
            If True, includes only float, int, boolean columns.
            If False, will raise error in-case there are
            non-numeric columns.

        Returns
        -------
        Series

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2, 3, 4, 5, 6, 6])
        >>> series
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        6    6
        dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 8, 10, 10]})
        >>> df.skew()
        a    0.00000
        b   -0.37037
        dtype: float64

        .. pandas-compat::
            **DataFrame.skew, Series.skew, Frame.skew**

            The `axis` parameter is not currently supported.
        """
        if axis not in (0, "index", None, no_default):
            raise NotImplementedError("Only axis=0 is currently supported.")

        return self._reduce(
            "skew",
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )

    @_cudf_nvtx_annotate
    def mask(self, cond, other=None, inplace: bool = False) -> Optional[Self]:
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is False, keep the original value.
            Where True, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is True are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.mask(df % 2 == 0, [-1, -1])
           A  B
        0  1  3
        1 -1  5
        2  5 -1

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.mask(ser > 2, 10)
        0    10
        1    10
        2     2
        3     1
        4     0
        dtype: int64
        >>> ser.mask(ser > 2)
        0    <NA>
        1    <NA>
        2       2
        3       1
        4       0
        dtype: int64
        """

        if not hasattr(cond, "__invert__"):
            # We Invert `cond` below and call `where`, so
            # making sure the object supports
            # `~`(inversion) operator or `__invert__` method
            cond = cp.asarray(cond)

        return self.where(cond=~cond, other=other, inplace=inplace)

    @_cudf_nvtx_annotate
    @copy_docstring(Rolling)
    def rolling(
        self, window, min_periods=None, center=False, axis=0, win_type=None
    ):
        return Rolling(
            self,
            window,
            min_periods=min_periods,
            center=center,
            axis=axis,
            win_type=win_type,
        )

    @_cudf_nvtx_annotate
    def nans_to_nulls(self):
        """
        Convert nans (if any) to nulls

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        **Series**

        >>> import cudf, numpy as np
        >>> series = cudf.Series([1, 2, np.nan, None, 10], nan_as_null=False)
        >>> series
        0     1.0
        1     2.0
        2     NaN
        3    <NA>
        4    10.0
        dtype: float64
        >>> series.nans_to_nulls()
        0     1.0
        1     2.0
        2    <NA>
        3    <NA>
        4    10.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame()
        >>> df['a'] = cudf.Series([1, None, np.nan], nan_as_null=False)
        >>> df['b'] = cudf.Series([None, 3.14, np.nan], nan_as_null=False)
        >>> df
              a     b
        0   1.0  <NA>
        1  <NA>  3.14
        2   NaN   NaN
        >>> df.nans_to_nulls()
              a     b
        0   1.0  <NA>
        1  <NA>  3.14
        2  <NA>  <NA>
        """
        result = (
            col.nans_to_nulls()
            if isinstance(col, cudf.core.column.NumericalColumn)
            else col.copy()
            for col in self._data.columns
        )
        return self._from_data_like_self(
            self._data._from_columns_like_self(result)
        )

    def _copy_type_metadata(
        self,
        other: Self,
        include_index: bool = True,
        *,
        override_dtypes: Optional[abc.Iterable[Optional[Dtype]]] = None,
    ) -> Self:
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.
        See `ColumnBase._with_type_metadata` for more information.
        """
        super()._copy_type_metadata(other, override_dtypes=override_dtypes)
        if (
            include_index
            and self._index is not None
            and other._index is not None
        ):
            self._index._copy_type_metadata(other._index)
            # When other._index is a CategoricalIndex, the current index
            # will be a NumericalIndex with an underlying CategoricalColumn
            # (the above _copy_type_metadata call will have converted the
            # column). Calling cudf.Index on that column generates the
            # appropriate index.
            if isinstance(
                other._index, cudf.core.index.CategoricalIndex
            ) and not isinstance(
                self._index, cudf.core.index.CategoricalIndex
            ):
                self._index = cudf.Index(
                    cast("cudf.Index", self._index)._column,
                    name=self._index.name,
                )
            elif isinstance(other._index, cudf.MultiIndex) and not isinstance(
                self._index, cudf.MultiIndex
            ):
                self._index = cudf.MultiIndex._from_data(
                    self._index._data, name=self._index.name
                )
        return self

    @_cudf_nvtx_annotate
    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction=None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        """
        Interpolate data values between some points.

        Parameters
        ----------
        method : str, default 'linear'
            Interpolation technique to use. Currently,
            only 'linear` is supported.
            * 'linear': Ignore the index and treat the values as
            equally spaced. This is the only method supported on MultiIndexes.
            * 'index', 'values': linearly interpolate using the index as
            an x-axis. Unsorted indices can lead to erroneous results.
        axis : int, default 0
            Axis to interpolate along. Currently,
            only 'axis=0' is supported.
        inplace : bool, default False
            Update the data in place if possible.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller, interpolated at
            some or all ``NaN`` values

        """
        if method in {"pad", "ffill"} and limit_direction != "forward":
            raise ValueError(
                f"`limit_direction` must be 'forward' for method `{method}`"
            )
        if method in {"backfill", "bfill"} and limit_direction != "backward":
            raise ValueError(
                f"`limit_direction` must be 'backward' for method `{method}`"
            )

        if method.lower() in {"ffill", "bfill", "pad", "backfill"}:
            warnings.warn(
                f"{type(self).__name__}.interpolate with method={method} is "
                "deprecated and will raise in a future version. "
                "Use obj.ffill() or obj.bfill() instead.",
                FutureWarning,
            )

        data = self

        if not isinstance(data._index, cudf.RangeIndex):
            perm_sort = data._index.argsort()
            data = data._gather(
                GatherMap.from_column_unchecked(
                    cudf.core.column.as_column(perm_sort),
                    len(data),
                    nullify=False,
                )
            )

        interpolator = cudf.core.algorithms.get_column_interpolator(method)
        columns = {}
        for colname, col in data._data.items():
            if isinstance(col, cudf.core.column.StringColumn):
                warnings.warn(
                    f"{type(self).__name__}.interpolate with object dtype is "
                    "deprecated and will raise in a future version.",
                    FutureWarning,
                )
            if col.nullable:
                col = col.astype("float64").fillna(np.nan)

            # Interpolation methods may or may not need the index
            columns[colname] = interpolator(col, index=data._index)

        result = self._from_data(columns, index=data._index)

        return (
            result
            if isinstance(data._index, cudf.RangeIndex)
            # TODO: This should be a scatter, avoiding an argsort.
            else result._gather(
                GatherMap.from_column_unchecked(
                    cudf.core.column.as_column(perm_sort.argsort()),
                    len(result),
                    nullify=False,
                )
            )
        )

    @_cudf_nvtx_annotate
    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values by `periods` positions."""
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise ValueError("Only axis=0 is supported.")
        if freq is not None:
            raise ValueError("The freq argument is not yet supported.")

        data_columns = (
            col.shift(periods, fill_value) for col in self._columns
        )
        return self.__class__._from_data(
            zip(self._column_names, data_columns), self._index
        )

    @_cudf_nvtx_annotate
    def truncate(self, before=None, after=None, axis=0, copy=True):
        """
        Truncate a Series or DataFrame before and after some index value.

        This is a useful shorthand for boolean indexing based on index
        values above or below certain thresholds.

        Parameters
        ----------
        before : date, str, int
            Truncate all rows before this index value.
        after : date, str, int
            Truncate all rows after this index value.
        axis : {0 or 'index', 1 or 'columns'}, optional
            Axis to truncate. Truncates the index (rows) by default.
        copy : bool, default is True,
            Return a copy of the truncated section.

        Returns
        -------
            The truncated Series or DataFrame.

        Notes
        -----
        If the index being truncated contains only datetime values,
        `before` and `after` may be specified as strings instead of
        Timestamps.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> cs1 = cudf.Series([1, 2, 3, 4])
        >>> cs1
        0    1
        1    2
        2    3
        3    4
        dtype: int64

        >>> cs1.truncate(before=1, after=2)
        1    2
        2    3
        dtype: int64

        >>> import cudf
        >>> dates = cudf.date_range(
        ...     '2021-01-01 23:45:00', '2021-01-01 23:46:00', freq='s'
        ... )
        >>> cs2 = cudf.Series(range(len(dates)), index=dates)
        >>> cs2
        2021-01-01 23:45:00     0
        2021-01-01 23:45:01     1
        2021-01-01 23:45:02     2
        2021-01-01 23:45:03     3
        2021-01-01 23:45:04     4
        2021-01-01 23:45:05     5
        2021-01-01 23:45:06     6
        2021-01-01 23:45:07     7
        2021-01-01 23:45:08     8
        2021-01-01 23:45:09     9
        2021-01-01 23:45:10    10
        2021-01-01 23:45:11    11
        2021-01-01 23:45:12    12
        2021-01-01 23:45:13    13
        2021-01-01 23:45:14    14
        2021-01-01 23:45:15    15
        2021-01-01 23:45:16    16
        2021-01-01 23:45:17    17
        2021-01-01 23:45:18    18
        2021-01-01 23:45:19    19
        2021-01-01 23:45:20    20
        2021-01-01 23:45:21    21
        2021-01-01 23:45:22    22
        2021-01-01 23:45:23    23
        2021-01-01 23:45:24    24
        ...
        2021-01-01 23:45:56    56
        2021-01-01 23:45:57    57
        2021-01-01 23:45:58    58
        2021-01-01 23:45:59    59
        dtype: int64


        >>> cs2.truncate(
        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ... )
        2021-01-01 23:45:18    18
        2021-01-01 23:45:19    19
        2021-01-01 23:45:20    20
        2021-01-01 23:45:21    21
        2021-01-01 23:45:22    22
        2021-01-01 23:45:23    23
        2021-01-01 23:45:24    24
        2021-01-01 23:45:25    25
        2021-01-01 23:45:26    26
        2021-01-01 23:45:27    27
        dtype: int64

        >>> cs3 = cudf.Series({'A': 1, 'B': 2, 'C': 3, 'D': 4})
        >>> cs3
        A    1
        B    2
        C    3
        D    4
        dtype: int64

        >>> cs3.truncate(before='B', after='C')
        B    2
        C    3
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({
        ...     'A': ['a', 'b', 'c', 'd', 'e'],
        ...     'B': ['f', 'g', 'h', 'i', 'j'],
        ...     'C': ['k', 'l', 'm', 'n', 'o']
        ... }, index=[1, 2, 3, 4, 5])
        >>> df
           A  B  C
        1  a  f  k
        2  b  g  l
        3  c  h  m
        4  d  i  n
        5  e  j  o

        >>> df.truncate(before=2, after=4)
           A  B  C
        2  b  g  l
        3  c  h  m
        4  d  i  n

        >>> df.truncate(before="A", after="B", axis="columns")
           A  B
        1  a  f
        2  b  g
        3  c  h
        4  d  i
        5  e  j

        >>> import cudf
        >>> dates = cudf.date_range(
        ...     '2021-01-01 23:45:00', '2021-01-01 23:46:00', freq='s'
        ... )
        >>> df2 = cudf.DataFrame(data={'A': 1, 'B': 2}, index=dates)
        >>> df2.head()
                             A  B
        2021-01-01 23:45:00  1  2
        2021-01-01 23:45:01  1  2
        2021-01-01 23:45:02  1  2
        2021-01-01 23:45:03  1  2
        2021-01-01 23:45:04  1  2

        >>> df2.truncate(
        ...     before="2021-01-01 23:45:18", after="2021-01-01 23:45:27"
        ... )
                             A  B
        2021-01-01 23:45:18  1  2
        2021-01-01 23:45:19  1  2
        2021-01-01 23:45:20  1  2
        2021-01-01 23:45:21  1  2
        2021-01-01 23:45:22  1  2
        2021-01-01 23:45:23  1  2
        2021-01-01 23:45:24  1  2
        2021-01-01 23:45:25  1  2
        2021-01-01 23:45:26  1  2
        2021-01-01 23:45:27  1  2

        .. pandas-compat::
            **DataFrame.truncate, Series.truncate**

            The ``copy`` parameter is only present for API compatibility, but
            ``copy=False`` is not supported. This method always generates a
            copy.
        """
        if not copy:
            raise ValueError("Truncating with copy=False is not supported.")
        axis = self._get_axis_from_axis_arg(axis)
        ax = self._index if axis == 0 else self._data.to_pandas_index()

        if not ax.is_monotonic_increasing and not ax.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")

        if type(ax) is cudf.core.index.DatetimeIndex:
            before = pd.to_datetime(before)
            after = pd.to_datetime(after)

        if before is not None and after is not None and before > after:
            raise ValueError(f"Truncate: {after} must be after {before}")

        if len(ax) > 1 and ax.is_monotonic_decreasing and ax.nunique() > 1:
            before, after = after, before

        slicer = [slice(None, None)] * self.ndim
        slicer[axis] = slice(before, after)
        return self.loc[tuple(slicer)].copy()

    @property
    def loc(self):
        """Select rows and columns by label or boolean mask.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([10, 11, 12], index=['a', 'b', 'c'])
        >>> series
        a    10
        b    11
        c    12
        dtype: int64
        >>> series.loc['b']
        11

        **DataFrame**

        DataFrame with string index.

        >>> df
           a  b
        a  0  5
        b  1  6
        c  2  7
        d  3  8
        e  4  9

        Select a single row by label.

        >>> df.loc['a']
        a    0
        b    5
        Name: a, dtype: int64

        Select multiple rows and a single column.

        >>> df.loc[['a', 'c', 'e'], 'b']
        a    5
        c    7
        e    9
        Name: b, dtype: int64

        Selection by boolean mask.

        >>> df.loc[df.a > 2]
           a  b
        d  3  8
        e  4  9

        Setting values using loc.

        >>> df.loc[['a', 'c', 'e'], 'a'] = 0
        >>> df
           a  b
        a  0  5
        b  1  6
        c  0  7
        d  3  8
        e  0  9

        """
        return self._loc_indexer_type(self)

    @property
    def iloc(self):
        """Select values by position.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> s = cudf.Series([10, 20, 30])
        >>> s
        0    10
        1    20
        2    30
        dtype: int64
        >>> s.iloc[2]
        30

        **DataFrame**

        Selecting rows and column by position.

        >>> df = cudf.DataFrame({'a': range(20),
        ...                      'b': range(20),
        ...                      'c': range(20)})

        Select a single row using an integer index.

        >>> df.iloc[1]
        a    1
        b    1
        c    1
        Name: 1, dtype: int64

        Select multiple rows using a list of integers.

        >>> df.iloc[[0, 2, 9, 18]]
              a    b    c
         0    0    0    0
         2    2    2    2
         9    9    9    9
        18   18   18   18

        Select rows using a slice.

        >>> df.iloc[3:10:2]
             a    b    c
        3    3    3    3
        5    5    5    5
        7    7    7    7
        9    9    9    9

        Select both rows and columns.

        >>> df.iloc[[1, 3, 5, 7], 2]
        1    1
        3    3
        5    5
        7    7
        Name: c, dtype: int64

        Setting values in a column using iloc.

        >>> df.iloc[:4] = 0
        >>> df
           a  b  c
        0  0  0  0
        1  0  0  0
        2  0  0  0
        3  0  0  0
        4  4  4  4
        5  5  5  5
        6  6  6  6
        7  7  7  7
        8  8  8  8
        9  9  9  9
        [10 more rows]

        """
        return self._iloc_indexer_type(self)

    @property  # type:ignore
    @_cudf_nvtx_annotate
    def axes(self):
        """
        Return a list representing the axes of the Series.

        Series.axes returns a list containing the row index.

        Examples
        --------
        >>> import cudf
        >>> csf1 = cudf.Series([1, 2, 3, 4])
        >>> csf1.axes
        [RangeIndex(start=0, stop=4, step=1)]

        """
        return [self.index]

    def squeeze(self, axis: Literal["index", "columns", 0, 1, None] = None):
        """
        Squeeze 1 dimensional axis objects into scalars.

        Series or DataFrames with a single element are squeezed to a scalar.
        DataFrames with a single column or a single row are squeezed to a
        Series. Otherwise the object is unchanged.

        This method is most useful when you don't know if your
        object is a Series or DataFrame, but you do know it has just a single
        column. In that case you can safely call `squeeze` to ensure you have a
        Series.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default None
            A specific axis to squeeze. By default, all length-1 axes are
            squeezed. For `Series` this parameter is unused and defaults
            to `None`.

        Returns
        -------
        DataFrame, Series, or scalar
            The projection after squeezing `axis` or all the axes.

        See Also
        --------
        Series.iloc : Integer-location based indexing for selecting scalars.
        DataFrame.iloc : Integer-location based indexing for selecting Series.
        Series.to_frame : Inverse of DataFrame.squeeze for a
            single-column DataFrame.

        Examples
        --------
        >>> primes = cudf.Series([2, 3, 5, 7])

        Slicing might produce a Series with a single value:

        >>> even_primes = primes[primes % 2 == 0]
        >>> even_primes
        0    2
        dtype: int64

        >>> even_primes.squeeze()
        2

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        dtype: int64

        Squeezing is even more effective when used with DataFrames.

        >>> df = cudf.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
        >>> df
           a  b
        0  1  2
        1  3  4

        Slicing a single column will produce a DataFrame with the columns
        having only one value:

        >>> df_a = df[["a"]]
        >>> df_a
           a
        0  1
        1  3

        So the columns can be squeezed down, resulting in a Series:

        >>> df_a.squeeze("columns")
        0    1
        1    3
        Name: a, dtype: int64

        Slicing a single row from a single column will produce a single
        scalar DataFrame:

        >>> df_0a = df.loc[df.index < 1, ["a"]]
        >>> df_0a
           a
        0  1

        Squeezing the rows produces a single scalar Series:

        >>> df_0a.squeeze("rows")
        a    1
        Name: 0, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_0a.squeeze()
        1
        """
        axes = (
            range(len(self.axes))
            if axis is None
            else (self._get_axis_from_axis_arg(axis),)
        )
        indexer = tuple(
            0 if i in axes and len(a) == 1 else slice(None)
            for i, a in enumerate(self.axes)
        )
        return self.iloc[indexer]

    @_cudf_nvtx_annotate
    def scale(self):
        """
        Scale values to [0, 1] in float64

        Returns
        -------
        DataFrame or Series
            Values scaled to [0, 1].

        Examples
        --------
        >>> import cudf
        >>> series = cudf.Series([10, 11, 12, 0.5, 1])
        >>> series
        0    10.0
        1    11.0
        2    12.0
        3     0.5
        4     1.0
        dtype: float64
        >>> series.scale()
        0    0.826087
        1    0.913043
        2    1.000000
        3    0.000000
        4    0.043478
        dtype: float64
        """
        vmin = self.min()
        vmax = self.max()
        scaled = (self - vmin) / (vmax - vmin)
        scaled._index = self._index.copy(deep=False)
        return scaled

    @_cudf_nvtx_annotate
    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind=None,
        na_position="last",
        sort_remaining=True,
        ignore_index=False,
        key=None,
    ):
        """Sort object by labels (along an axis).

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis along which to sort. The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
            This is only useful in the case of MultiIndex.
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : sorting method such as `quick sort` and others.
            Not yet supported.
        na_position : {'first', 'last'}, default 'last'
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        sort_remaining : bool, default True
            When sorting a multiindex on a subset of its levels,
            should entries be lexsorted by the remaining
            (non-specified) levels as well?
        ignore_index : bool, default False
            if True, index will be replaced with RangeIndex.
        key : callable, optional
            If not None, apply the key function to the index values before
            sorting. This is similar to the key argument in the builtin
            sorted() function, with the notable difference that this key
            function should be vectorized. It should expect an Index and return
            an Index of the same shape. For MultiIndex inputs, the key is
            applied per level.

        Returns
        -------
        Frame or None

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series(['a', 'b', 'c', 'd'], index=[3, 2, 1, 4])
        >>> series
        3    a
        2    b
        1    c
        4    d
        dtype: object
        >>> series.sort_index()
        1    c
        2    b
        3    a
        4    d
        dtype: object

        Sort Descending

        >>> series.sort_index(ascending=False)
        4    d
        3    a
        2    b
        1    c
        dtype: object

        **DataFrame**

        >>> df = cudf.DataFrame(
        ... {"b":[3, 2, 1], "a":[2, 1, 3]}, index=[1, 3, 2])
        >>> df.sort_index(axis=0)
           b  a
        1  3  2
        2  1  3
        3  2  1
        >>> df.sort_index(axis=1)
           a  b
        1  2  3
        3  1  2
        2  3  1

        .. pandas-compat::
            **DataFrame.sort_index, Series.sort_index**

            * Not supporting: kind, sort_remaining=False
        """
        if kind is not None:
            raise NotImplementedError("kind is not yet supported")

        if key is not None:
            raise NotImplementedError("key is not yet supported.")

        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        if axis in (0, "index"):
            idx = self.index
            if isinstance(idx, MultiIndex):
                if level is not None:
                    if not is_list_like(level):
                        level = [level]
                    by = list(map(idx._get_level_label, level))
                    if sort_remaining:
                        handled = set(by)
                        by.extend(
                            filter(
                                lambda n: n not in handled,
                                self.index._data.names,
                            )
                        )
                else:
                    by = list(idx._data.names)

                inds = idx._get_sorted_inds(
                    by=by, ascending=ascending, na_position=na_position
                )
                out = self._gather(
                    GatherMap.from_column_unchecked(
                        inds, len(self), nullify=False
                    )
                )
                # TODO: frame factory function should handle multilevel column
                # names
                if (
                    isinstance(self, cudf.core.dataframe.DataFrame)
                    and self._data.multiindex
                ):
                    out._set_columns_like(self._data)
            elif (ascending and idx.is_monotonic_increasing) or (
                not ascending and idx.is_monotonic_decreasing
            ):
                out = self.copy()
            else:
                inds = idx.argsort(
                    ascending=ascending, na_position=na_position
                )
                out = self._gather(
                    GatherMap.from_column_unchecked(
                        cudf.core.column.as_column(inds),
                        len(self),
                        nullify=False,
                    )
                )
                if (
                    isinstance(self, cudf.core.dataframe.DataFrame)
                    and self._data.multiindex
                ):
                    out._set_columns_like(self._data)
            if ignore_index:
                out = out.reset_index(drop=True)
        else:
            labels = sorted(self._data.names, reverse=not ascending)
            out = self[labels]
            if ignore_index:
                out._data.rangeindex = True
                out._data.names = list(range(len(self._data.names)))

        return self._mimic_inplace(out, inplace=inplace)

    def memory_usage(self, index=True, deep=False):
        """Return the memory usage of an object.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the index.
        deep : bool, default False
            The deep parameter is ignored and is only included for pandas
            compatibility.

        Returns
        -------
        Series or scalar
            For DataFrame, a Series whose index is the original column names
            and whose values is the memory usage of each column in bytes. For a
            Series the total memory usage.

        Examples
        --------
        **DataFrame**

        >>> dtypes = ['int64', 'float64', 'object', 'bool']
        >>> data = dict([(t, np.ones(shape=5000).astype(t))
        ...              for t in dtypes])
        >>> df = cudf.DataFrame(data)
        >>> df.head()
           int64  float64  object  bool
        0      1      1.0     1.0  True
        1      1      1.0     1.0  True
        2      1      1.0     1.0  True
        3      1      1.0     1.0  True
        4      1      1.0     1.0  True
        >>> df.memory_usage(index=False)
        int64      40000
        float64    40000
        object     40000
        bool        5000
        dtype: int64

        Use a Categorical for efficient storage of an object-dtype column with
        many repeated values.

        >>> df['object'].astype('category').memory_usage(deep=True)
        5008

        **Series**
        >>> s = cudf.Series(range(3), index=['a','b','c'])
        >>> s.memory_usage()
        43

        Not including the index gives the size of the rest of the data, which
        is necessarily smaller:

        >>> s.memory_usage(index=False)
        24
        """
        raise NotImplementedError

    def hash_values(self, method="murmur3", seed=None):
        """Compute the hash of values in this column.

        Parameters
        ----------
        method : {'murmur3', 'md5', 'xxhash64'}, default 'murmur3'
            Hash function to use:

            * murmur3: MurmurHash3 hash function
            * md5: MD5 hash function
            * xxhash64: xxHash64 hash function

        seed : int, optional
            Seed value to use for the hash function. This parameter is only
            supported for 'murmur3' and 'xxhash64'.


        Returns
        -------
        Series
            A Series with hash values.

        Examples
        --------
        **Series**

        >>> import cudf
        >>> series = cudf.Series([10, 120, 30])
        >>> series
        0     10
        1    120
        2     30
        dtype: int64
        >>> series.hash_values(method="murmur3")
        0   -1930516747
        1     422619251
        2    -941520876
        dtype: int32
        >>> series.hash_values(method="md5")
        0    7be4bbacbfdb05fb3044e36c22b41e8b
        1    947ca8d2c5f0f27437f156cfbfab0969
        2    d0580ef52d27c043c8e341fd5039b166
        dtype: object
        >>> series.hash_values(method="murmur3", seed=42)
        0    2364453205
        1     422621911
        2    3353449140
        dtype: uint32

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({"a": [10, 120, 30], "b": [0.0, 0.25, 0.50]})
        >>> df
             a     b
        0   10  0.00
        1  120  0.25
        2   30  0.50
        >>> df.hash_values(method="murmur3")
        0    -330519225
        1    -397962448
        2   -1345834934
        dtype: int32
        >>> df.hash_values(method="md5")
        0    57ce879751b5169c525907d5c563fae1
        1    948d6221a7c4963d4be411bcead7e32b
        2    fe061786ea286a515b772d91b0dfcd70
        dtype: object
        """
        seed_hash_methods = {"murmur3", "xxhash64"}
        if seed is None:
            seed = 0
        elif method not in seed_hash_methods:
            warnings.warn(
                "Provided seed value has no effect for the hash method "
                f"`{method}`. Only {seed_hash_methods} support seeds."
            )
        # Note that both Series and DataFrame return Series objects from this
        # calculation, necessitating the unfortunate circular reference to the
        # child class here.
        return cudf.Series._from_data(
            {None: libcudf.hash.hash([*self._columns], method, seed)},
            index=self.index,
        )

    def _gather(
        self,
        gather_map: GatherMap,
        keep_index=True,
    ):
        """Gather rows of frame specified by indices in `gather_map`.

        Maintain the index if keep_index is True.

        This function does no expensive bounds checking, but does
        check that the number of rows of self matches the validated
        number of rows.
        """
        if not gather_map.nullify and len(self) != gather_map.nrows:
            raise IndexError("Gather map is out of bounds")
        return self._from_columns_like_self(
            libcudf.copying.gather(
                list(self._index._columns + self._columns)
                if keep_index
                else list(self._columns),
                gather_map.column,
                nullify=gather_map.nullify,
            ),
            self._column_names,
            self._index.names if keep_index else None,
        )

    def _slice(self, arg: slice, keep_index: bool = True) -> Self:
        """Slice a frame.

        Parameters
        ----------
        arg
            The slice
        keep_index
            Preserve the index when slicing?

        Returns
        -------
        Sliced frame

        Notes
        -----
        This slicing has normal python semantics.
        """
        num_rows = len(self)
        if num_rows == 0:
            return self
        start, stop, stride = arg.indices(num_rows)
        index = self.index
        has_range_index = isinstance(index, RangeIndex)
        if len(range(start, stop, stride)) == 0:
            # Avoid materialising the range index column
            result = self._empty_like(
                keep_index=keep_index and not has_range_index
            )
            if keep_index and has_range_index:
                lo = index.start + start * index.step
                hi = index.start + stop * index.step
                step = index.step * stride
                result.index = RangeIndex(
                    start=lo, stop=hi, step=step, name=index.name
                )
            return result
        if start < 0:
            start = start + num_rows

        # At this point, we have converted slice arguments into
        # indices that no longer wrap around.
        # For example slice(4, None, -1) will produce the
        # start, stop, stride tuple (4, -1, -1)
        # This check makes sure -1 is not wrapped (again) to
        # produce -1 + num_rows.
        if stop < 0 and not (stride < 0 and stop == -1):
            stop = stop + num_rows
        stride = 1 if stride is None else stride

        if (stop - start) * stride <= 0:
            return self._empty_like(keep_index=True)

        start = min(start, num_rows)
        stop = min(stop, num_rows)

        if stride != 1:
            return self._gather(
                GatherMap.from_column_unchecked(
                    as_column(
                        range(start, stop, stride),
                        dtype=libcudf.types.size_type_dtype,
                    ),
                    len(self),
                    nullify=False,
                ),
                keep_index=keep_index,
            )

        columns_to_slice = [
            *(
                self._index._data.columns
                if keep_index and not has_range_index
                else []
            ),
            *self._columns,
        ]
        result = self._from_columns_like_self(
            libcudf.copying.columns_slice(columns_to_slice, [start, stop])[0],
            self._column_names,
            None if has_range_index or not keep_index else self._index.names,
        )
        result._data.label_dtype = self._data.label_dtype
        result._data.rangeindex = self._data.rangeindex

        if keep_index and has_range_index:
            result.index = self.index[start:stop]
        return result

    def _positions_from_column_names(
        self, column_names, offset_by_index_columns=False
    ):
        """Map each column name into their positions in the frame.

        Return positions of the provided column names, offset by the number of
        index columns if `offset_by_index_columns` is True. The order of
        indices returned corresponds to the column order in this Frame.
        """
        num_index_columns = (
            len(self._index._data) if offset_by_index_columns else 0
        )
        return [
            i + num_index_columns
            for i, name in enumerate(self._column_names)
            if name in set(column_names)
        ]

    def drop_duplicates(
        self,
        subset=None,
        keep="first",
        nulls_are_equal=True,
        ignore_index=False,
    ):
        """
        Drop duplicate rows in frame.

        subset : list, optional
            List of columns to consider when dropping rows.
        keep : ["first", "last", False]
            "first" will keep the first duplicate entry, "last" will keep the
            last duplicate entry, and False will drop all duplicates.
        nulls_are_equal: bool, default True
            Null elements are considered equal to other null elements.
        ignore_index: bool, default False
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.
        """
        if not isinstance(ignore_index, (np.bool_, bool)):
            raise ValueError(
                f"{ignore_index=} must be bool, "
                f"not {type(ignore_index).__name__}"
            )
        subset = self._preprocess_subset(subset)
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        keys = self._positions_from_column_names(
            subset, offset_by_index_columns=not ignore_index
        )
        return self._from_columns_like_self(
            libcudf.stream_compaction.drop_duplicates(
                list(self._columns)
                if ignore_index
                else list(self._index._columns + self._columns),
                keys=keys,
                keep=keep,
                nulls_are_equal=nulls_are_equal,
            ),
            self._column_names,
            self._index.names if not ignore_index else None,
        )

    @_cudf_nvtx_annotate
    def duplicated(self, subset=None, keep="first"):
        """
        Return boolean Series denoting duplicate rows.

        Considering certain columns is optional.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to mark.

            - ``'first'`` : Mark duplicates as ``True`` except for the first
                occurrence.
            - ``'last'`` : Mark duplicates as ``True`` except for the last
                occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        Series
            Boolean series indicating duplicated rows.

        See Also
        --------
        Index.duplicated : Equivalent method on index.
        Series.duplicated : Equivalent method on Series.
        Series.drop_duplicates : Remove duplicate values from Series.
        DataFrame.drop_duplicates : Remove duplicate values from DataFrame.

        Examples
        --------
        Consider a dataset containing ramen product ratings.

        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Maggie', 'Maggie', 'Maggie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
             brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2   Maggie   cup     3.5
        3   Maggie  pack    15.0
        4   Maggie  pack     5.0

        By default, for each set of duplicated values, the first occurrence
        is set to False and all others to True.

        >>> df.duplicated()
        0    False
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        By using 'last', the last occurrence of each set of duplicated values
        is set to False and all others to True.

        >>> df.duplicated(keep='last')
        0     True
        1    False
        2    False
        3    False
        4    False
        dtype: bool

        By setting ``keep`` to False, all duplicates are True.

        >>> df.duplicated(keep=False)
        0     True
        1     True
        2    False
        3    False
        4    False
        dtype: bool

        To find duplicates on specific column(s), use ``subset``.

        >>> df.duplicated(subset=['brand'])
        0    False
        1     True
        2    False
        3     True
        4     True
        dtype: bool
        """
        subset = self._preprocess_subset(subset)

        if isinstance(self, cudf.Series):
            columns = [self._column]
        else:
            columns = [self._data[n] for n in subset]
        distinct = libcudf.stream_compaction.distinct_indices(
            columns, keep=keep
        )
        (result,) = libcudf.copying.scatter(
            [cudf.Scalar(False, dtype=bool)],
            distinct,
            [as_column(True, length=len(self), dtype=bool)],
            bounds_check=False,
        )
        return cudf.Series(result, index=self.index)

    @_cudf_nvtx_annotate
    def _empty_like(self, keep_index=True) -> Self:
        result = self._from_columns_like_self(
            libcudf.copying.columns_empty_like(
                [
                    *(self._index._data.columns if keep_index else ()),
                    *self._columns,
                ]
            ),
            self._column_names,
            self._index.names if keep_index else None,
        )
        result._data.label_dtype = self._data.label_dtype
        result._data.rangeindex = self._data.rangeindex
        return result

    def _split(self, splits, keep_index=True):
        if self._num_rows == 0:
            return []

        columns_split = libcudf.copying.columns_split(
            [
                *(self._index._data.columns if keep_index else []),
                *self._columns,
            ],
            splits,
        )

        return [
            self._from_columns_like_self(
                columns_split[i],
                self._column_names,
                self._index.names if keep_index else None,
            )
            for i in range(len(splits) + 1)
        ]

    @_cudf_nvtx_annotate
    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ):  # noqa: D102
        if method is not None:
            # Do not remove until pandas 3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.warn(
                f"{type(self).__name__}.fillna with 'method' is "
                "deprecated and will raise in a future version. "
                "Use obj.ffill() or obj.bfill() instead.",
                FutureWarning,
            )
        old_index = self._index
        ret = super().fillna(value, method, axis, inplace, limit)
        if inplace:
            self._index = old_index
        else:
            ret._index = old_index
        return ret

    @_cudf_nvtx_annotate
    def bfill(self, value=None, axis=None, inplace=None, limit=None):
        """
        Synonym for :meth:`Series.fillna` with ``method='bfill'``.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return self.fillna(
                method="bfill",
                value=value,
                axis=axis,
                inplace=inplace,
                limit=limit,
            )

    @_cudf_nvtx_annotate
    def backfill(self, value=None, axis=None, inplace=None, limit=None):
        """
        Synonym for :meth:`Series.fillna` with ``method='bfill'``.

        .. deprecated:: 23.06
           Use `DataFrame.bfill/Series.bfill` instead.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            "DataFrame.backfill/Series.backfill is deprecated. Use "
            "DataFrame.bfill/Series.bfill instead",
            FutureWarning,
        )
        return self.bfill(value=value, axis=axis, inplace=inplace, limit=limit)

    @_cudf_nvtx_annotate
    def ffill(self, value=None, axis=None, inplace=None, limit=None):
        """
        Synonym for :meth:`Series.fillna` with ``method='ffill'``.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return self.fillna(
                method="ffill",
                value=value,
                axis=axis,
                inplace=inplace,
                limit=limit,
            )

    @_cudf_nvtx_annotate
    def pad(self, value=None, axis=None, inplace=None, limit=None):
        """
        Synonym for :meth:`Series.fillna` with ``method='ffill'``.

        .. deprecated:: 23.06
           Use `DataFrame.ffill/Series.ffill` instead.

        Returns
        -------
            Object with missing values filled or None if ``inplace=True``.
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            "DataFrame.pad/Series.pad is deprecated. Use "
            "DataFrame.ffill/Series.ffill instead",
            FutureWarning,
        )
        return self.ffill(value=value, axis=axis, inplace=inplace, limit=limit)

    def add_prefix(self, prefix):
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
            The string to add before each label.

        Returns
        -------
        Series or DataFrame
            New Series with updated labels or DataFrame with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix row labels with string 'suffix'.
        DataFrame.add_suffix: Suffix column labels with string 'suffix'.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> s.add_prefix('item_')
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6
        >>> df.add_prefix('col_')
             col_A  col_B
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        raise NotImplementedError(
            "`IndexedFrame.add_prefix` not currently implemented. \
                Use `Series.add_prefix` or `DataFrame.add_prefix`"
        )

    def add_suffix(self, suffix):
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        prefix : str
            The string to add after each label.

        Returns
        -------
        Series or DataFrame
            New Series with updated labels or DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: prefix row labels with string 'prefix'.
        DataFrame.add_prefix: Prefix column labels with string 'prefix'.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        dtype: int64
        >>> s.add_suffix('_item')
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        dtype: int64

        **DataFrame**

        >>> df = cudf.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6
        >>> df.add_suffix('_col')
             A_col  B_col
        0       1       3
        1       2       4
        2       3       5
        3       4       6
        """
        raise NotImplementedError

    @acquire_spill_lock()
    @_cudf_nvtx_annotate
    def _apply(self, func, kernel_getter, *args, **kwargs):
        """Apply `func` across the rows of the frame."""
        if kwargs:
            raise ValueError("UDFs using **kwargs are not yet supported.")
        try:
            kernel, retty = _compile_or_get(
                self, func, args, kernel_getter=kernel_getter
            )
        except Exception as e:
            raise ValueError(
                "user defined function compilation failed."
            ) from e

        # Mask and data column preallocated
        ans_col = _return_arr_from_dtype(retty, len(self))
        ans_mask = as_column(True, length=len(self), dtype="bool")
        output_args = [(ans_col, ans_mask), len(self)]
        input_args = _get_input_args_from_frame(self)
        launch_args = output_args + input_args + list(args)
        try:
            with _CUDFNumbaConfig():
                kernel.forall(len(self))(*launch_args)
        except Exception as e:
            raise RuntimeError("UDF kernel execution failed.") from e

        col = _post_process_output_col(ans_col, retty)

        col.set_base_mask(libcudf.transform.bools_to_mask(ans_mask))
        result = cudf.Series._from_data({None: col}, self._index)

        return result

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
        """Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        na_position : {'first', 'last'}, default 'last'
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.

        Returns
        -------
        Frame : Frame with sorted values.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['a'] = [0, 1, 2]
        >>> df['b'] = [-3, 2, 0]
        >>> df.sort_values('b')
           a  b
        0  0 -3
        2  2  0
        1  1  2

        .. pandas-compat::
            **DataFrame.sort_values, Series.sort_values**

            * Support axis='index' only.
            * Not supporting: inplace, kind
        """
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            if kind not in {"mergesort", "heapsort", "stable"}:
                raise AttributeError(
                    f"{kind} is not a valid sorting algorithm for "
                    f"'DataFrame' object"
                )
            warnings.warn(
                f"GPU-accelerated {kind} is currently not supported, "
                f"defaulting to quicksort."
            )
        if axis != 0:
            raise NotImplementedError("`axis` not currently implemented.")

        if len(self) == 0:
            return self

        # argsort the `by` column
        out = self._gather(
            GatherMap.from_column_unchecked(
                self._get_columns_by_label(by)._get_sorted_inds(
                    ascending=ascending, na_position=na_position
                ),
                len(self),
                nullify=False,
            ),
            keep_index=not ignore_index,
        )
        if (
            isinstance(self, cudf.core.dataframe.DataFrame)
            and self._data.multiindex
        ):
            out.columns = self._data.to_pandas_index()
        return out

    def _n_largest_or_smallest(
        self, largest: bool, n: int, columns, keep: Literal["first", "last"]
    ):
        # Get column to operate on
        if isinstance(columns, str):
            columns = [columns]

        method = "nlargest" if largest else "nsmallest"
        for col in columns:
            if isinstance(self._data[col], cudf.core.column.StringColumn):
                if isinstance(self, cudf.DataFrame):
                    error_msg = (
                        f"Column '{col}' has dtype {self._data[col].dtype}, "
                        f"cannot use method '{method}' with this dtype"
                    )
                else:
                    error_msg = (
                        f"Cannot use method '{method}' with "
                        f"dtype {self._data[col].dtype}"
                    )
                raise TypeError(error_msg)
        if len(self) == 0:
            return self

        if keep == "first":
            if n < 0:
                n = 0

            # argsort the `by` column
            return self._gather(
                GatherMap.from_column_unchecked(
                    self._get_columns_by_label(columns)
                    ._get_sorted_inds(ascending=not largest)
                    .slice(*slice(None, n).indices(len(self))),
                    len(self),
                    nullify=False,
                ),
                keep_index=True,
            )
        elif keep == "last":
            indices = self._get_columns_by_label(columns)._get_sorted_inds(
                ascending=largest
            )

            if n <= 0:
                # Empty slice.
                indices = indices.slice(0, 0)
            else:
                indices = indices.slice(
                    *slice(None, -n - 1, -1).indices(len(self))
                )
            return self._gather(
                GatherMap.from_column_unchecked(
                    indices, len(self), nullify=False
                ),
                keep_index=True,
            )
        else:
            raise ValueError('keep must be either "first", "last"')

    def _align_to_index(
        self,
        index: ColumnLike,
        how: str = "outer",
        sort: bool = True,
        allow_non_unique: bool = False,
    ) -> Self:
        index = cudf.core.index.as_index(index)

        if self.index.equals(index):
            return self
        if not allow_non_unique:
            if not self.index.is_unique or not index.is_unique:
                raise ValueError("Cannot align indices with non-unique values")

        lhs = cudf.DataFrame._from_data(self._data, index=self.index)
        rhs = cudf.DataFrame._from_data({}, index=index)

        # create a temporary column that we will later sort by
        # to recover ordering after index alignment.
        sort_col_id = str(uuid4())
        if how == "left":
            lhs[sort_col_id] = as_column(range(len(lhs)))
        elif how == "right":
            rhs[sort_col_id] = as_column(range(len(rhs)))

        result = lhs.join(rhs, how=how, sort=sort)
        if how in ("left", "right"):
            result = result.sort_values(sort_col_id)
            del result[sort_col_id]

        result = self.__class__._from_data(
            data=result._data, index=result.index
        )
        result._data.multiindex = self._data.multiindex
        result._data._level_names = self._data._level_names
        result.index.names = self.index.names

        return result

    @_cudf_nvtx_annotate
    def _reindex(
        self,
        column_names,
        dtypes=None,
        deep=False,
        index=None,
        inplace=False,
        fill_value=NA,
    ):
        """
        Helper for `.reindex`

        Parameters
        ----------
        columns_names : array-like
            array-like of columns to select from the Frame,
            if ``columns`` is a superset of ``Frame.columns`` new
            columns are created.
        dtypes : dict
            Mapping of dtypes for the empty columns being created.
        deep : boolean, optional, default False
            Whether to make deep copy or shallow copy of the columns.
        index : Index or array-like, default None
            The ``index`` to be used to reindex the Frame with.
        inplace : bool, default False
            Whether to perform the operation in place on the data.
        fill_value : value with which to replace nulls in the result

        Returns
        -------
        Series or DataFrame
        """
        if dtypes is None:
            dtypes = {}

        df = self
        if index is not None:
            if not df._index.is_unique:
                raise ValueError(
                    "cannot reindex on an axis with duplicate labels"
                )
            index = cudf.core.index.as_index(
                index, name=getattr(index, "name", self._index.name)
            )

            idx_dtype_match = (df.index.nlevels == index.nlevels) and all(
                _is_same_dtype(left_dtype, right_dtype)
                for left_dtype, right_dtype in zip(
                    (col.dtype for col in df.index._data.columns),
                    (col.dtype for col in index._data.columns),
                )
            )

            if not idx_dtype_match:
                column_names = (
                    column_names
                    if column_names is not None
                    else list(df._column_names)
                )
                df = cudf.DataFrame()
            else:
                lhs = cudf.DataFrame._from_data({}, index=index)
                rhs = cudf.DataFrame._from_data(
                    {
                        # bookkeeping workaround for unnamed series
                        (name or 0)
                        if isinstance(self, cudf.Series)
                        else name: col
                        for name, col in df._data.items()
                    },
                    index=df._index,
                )
                df = lhs.join(rhs, how="left", sort=True)
                # double-argsort to map back from sorted to unsorted positions
                df = df.take(index.argsort(ascending=True).argsort())

        index = index if index is not None else df.index

        if column_names is None:
            names = list(df._data.names)
            level_names = self._data.level_names
            multiindex = self._data.multiindex
            rangeindex = self._data.rangeindex
        elif isinstance(column_names, (pd.Index, cudf.Index)):
            if isinstance(column_names, (pd.MultiIndex, cudf.MultiIndex)):
                multiindex = True
                if isinstance(column_names, cudf.MultiIndex):
                    names = list(iter(column_names.to_pandas()))
                else:
                    names = list(iter(column_names))
                rangeindex = False
            else:
                multiindex = False
                names = column_names
                if isinstance(names, cudf.Index):
                    names = names.to_pandas()
                rangeindex = isinstance(
                    column_names, (pd.RangeIndex, cudf.RangeIndex)
                )
            level_names = tuple(column_names.names)
        else:
            names = column_names
            level_names = None
            multiindex = False
            rangeindex = False

        cols = {
            name: (
                df._data[name].copy(deep=deep)
                if name in df._data
                else cudf.core.column.column.column_empty(
                    dtype=dtypes.get(name, np.float64),
                    masked=True,
                    row_count=len(index),
                )
            )
            for name in names
        }

        result = self.__class__._from_data(
            data=cudf.core.column_accessor.ColumnAccessor(
                cols,
                multiindex=multiindex,
                level_names=level_names,
                rangeindex=rangeindex,
            ),
            index=index,
        )

        result.fillna(fill_value, inplace=True)
        return self._mimic_inplace(result, inplace=inplace)

    def round(self, decimals=0, how="half_even"):
        """
        Round to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. This parameter
            must be an int for a Series. For a DataFrame, a dict or a Series
            are also valid inputs. If an int is given, round each column to the
            same number of places. Otherwise dict and Series round to variable
            numbers of places. Column names should be in the keys if
            `decimals` is a dict-like, or in the index if `decimals` is a
            Series. Any columns not included in `decimals` will be left as is.
            Elements of `decimals` which are not columns of the input will be
            ignored.
        how : str, optional
            Type of rounding. Can be either "half_even" (default)
            or "half_up" rounding.

        Returns
        -------
        Series or DataFrame
            A Series or DataFrame with the affected columns rounded to the
            specified number of decimal places.

        Examples
        --------
        **Series**

        >>> s = cudf.Series([0.1, 1.4, 2.9])
        >>> s.round()
        0    0.0
        1    1.0
        2    3.0
        dtype: float64

        **DataFrame**

        >>> df = cudf.DataFrame(
        ...     [(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...     columns=['dogs', 'cats'],
        ... )
        >>> df
           dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places.

        >>> df.round(1)
           dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as keys and the number of decimal
        places as values.

        >>> df.round({'dogs': 1, 'cats': 0})
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as the index and the number of
        decimal places as the values.

        >>> decimals = cudf.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
           dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """
        if isinstance(decimals, cudf.Series):
            decimals = decimals.to_pandas()

        if isinstance(decimals, pd.Series):
            if not decimals.index.is_unique:
                raise ValueError("Index of decimals must be unique")
            decimals = decimals.to_dict()
        elif isinstance(decimals, int):
            decimals = {name: decimals for name in self._column_names}
        elif not isinstance(decimals, abc.Mapping):
            raise TypeError(
                "decimals must be an integer, a dict-like or a Series"
            )

        cols = {
            name: col.round(decimals[name], how=how)
            if (
                name in decimals
                and _is_non_decimal_numeric_dtype(col.dtype)
                and not is_bool_dtype(col.dtype)
            )
            else col.copy(deep=True)
            for name, col in self._data.items()
        }

        return self.__class__._from_data(
            data=cudf.core.column_accessor.ColumnAccessor(
                cols,
                multiindex=self._data.multiindex,
                level_names=self._data.level_names,
            ),
            index=self._index,
        )

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=None,
        on=None,
        level=None,
        origin="start_day",
        offset=None,
    ):
        """
        Convert the frequency of ("resample") the given time series data.

        Parameters
        ----------
        rule: str
            The offset string representing the frequency to use.
            Note that DateOffset objects are not yet supported.
        closed: {"right", "left"}, default None
            Which side of bin interval is closed. The default is
            "left" for all frequency offsets except for "M" and "W",
            which have a default of "right".
        label: {"right", "left"}, default None
            Which bin edge label to label bucket with. The default is
            "left" for all frequency offsets except for "M" and "W",
            which have a default of "right".
        on: str, optional
            For a DataFrame, column to use instead of the index for
            resampling.  Column must be a datetime-like.
        level: str or int, optional
            For a MultiIndex, level to use instead of the index for
            resampling.  The level must be a datetime-like.

        Returns
        -------
        A Resampler object

        Examples
        --------
        First, we create a time series with 1 minute intervals:

        >>> index = cudf.date_range(start="2001-01-01", periods=10, freq="1T")
        >>> sr = cudf.Series(range(10), index=index)
        >>> sr
        2001-01-01 00:00:00    0
        2001-01-01 00:01:00    1
        2001-01-01 00:02:00    2
        2001-01-01 00:03:00    3
        2001-01-01 00:04:00    4
        2001-01-01 00:05:00    5
        2001-01-01 00:06:00    6
        2001-01-01 00:07:00    7
        2001-01-01 00:08:00    8
        2001-01-01 00:09:00    9
        dtype: int64

        Downsampling to 3 minute intervals, followed by a "sum" aggregation:

        >>> sr.resample("3T").sum()
        2001-01-01 00:00:00     3
        2001-01-01 00:03:00    12
        2001-01-01 00:06:00    21
        2001-01-01 00:09:00     9
        dtype: int64

        Use the right side of each interval to label the bins:

        >>> sr.resample("3T", label="right").sum()
        2001-01-01 00:03:00     3
        2001-01-01 00:06:00    12
        2001-01-01 00:09:00    21
        2001-01-01 00:12:00     9
        dtype: int64

        Close the right side of the interval instead of the left:

        >>> sr.resample("3T", closed="right").sum()
        2000-12-31 23:57:00     0
        2001-01-01 00:00:00     6
        2001-01-01 00:03:00    15
        2001-01-01 00:06:00    24
        dtype: int64

        Upsampling to 30 second intervals:

        >>> sr.resample("30s").asfreq()[:5]  # show the first 5 rows
        2001-01-01 00:00:00       0
        2001-01-01 00:00:30    <NA>
        2001-01-01 00:01:00       1
        2001-01-01 00:01:30    <NA>
        2001-01-01 00:02:00       2
        dtype: int64

        Upsample and fill nulls using the "bfill" method:

        >>> sr.resample("30s").bfill()[:5]
        2001-01-01 00:00:00    0
        2001-01-01 00:00:30    1
        2001-01-01 00:01:00    1
        2001-01-01 00:01:30    2
        2001-01-01 00:02:00    2
        dtype: int64

        Resampling by a specified column of a Dataframe:

        >>> df = cudf.DataFrame({
        ...     "price": [10, 11, 9, 13, 14, 18, 17, 19],
        ...     "volume": [50, 60, 40, 100, 50, 100, 40, 50],
        ...     "week_starting": cudf.date_range(
        ...         "2018-01-01", periods=8, freq="7D"
        ...     )
        ... })
        >>> df
        price  volume week_starting
        0     10      50    2018-01-01
        1     11      60    2018-01-08
        2      9      40    2018-01-15
        3     13     100    2018-01-22
        4     14      50    2018-01-29
        5     18     100    2018-02-05
        6     17      40    2018-02-12
        7     19      50    2018-02-19
        >>> df.resample("M", on="week_starting").mean()
                       price     volume
        week_starting
        2018-01-31      11.4  60.000000
        2018-02-28      18.0  63.333333


        .. pandas-compat::
            **DataFrame.resample, Series.resample**

            Note that the dtype of the index (or the 'on' column if using
            'on=') in the result will be of a frequency closest to the
            resampled frequency.  For example, if resampling from
            nanoseconds to milliseconds, the index will be of dtype
            'datetime64[ms]'.
        """
        import cudf.core.resample

        if kind is not None:
            warnings.warn(
                "The 'kind' keyword in is "
                "deprecated and will be removed in a future version. ",
                FutureWarning,
            )
        if (axis, convention, kind, loffset, base, origin, offset) != (
            0,
            "start",
            None,
            None,
            None,
            "start_day",
            None,
        ):
            raise NotImplementedError(
                "The following arguments are not "
                "currently supported by resample:\n\n"
                "- axis\n"
                "- convention\n"
                "- kind\n"
                "- loffset\n"
                "- base\n"
                "- origin\n"
                "- offset"
            )
        by = cudf.Grouper(
            key=on, freq=rule, closed=closed, label=label, level=level
        )
        return (
            cudf.core.resample.SeriesResampler(self, by=by)
            if isinstance(self, cudf.Series)
            else cudf.core.resample.DataFrameResampler(self, by=by)
        )

    def dropna(
        self, axis=0, how="any", thresh=None, subset=None, inplace=False
    ):
        """
        Drop rows (or columns) containing nulls from a Column.

        Parameters
        ----------
        axis : {0, 1}, optional
            Whether to drop rows (axis=0, default) or columns (axis=1)
            containing nulls.
        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row (or column).
            any (default) drops rows (or columns) containing at least
            one null value. all drops only rows (or columns) containing
            *all* null values.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.

        See Also
        --------
        cudf.DataFrame.isna
            Indicate null values.
        cudf.DataFrame.notna
            Indicate non-null values.
        cudf.DataFrame.fillna
            Replace null values.
        cudf.Series.dropna
            Drop null values.
        cudf.Index.dropna
            Drop null indices.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": ['Batmobile', None, 'Bullwhip'],
        ...                    "born": [np.datetime64("1940-04-25"),
        ...                             np.datetime64("NaT"),
        ...                             np.datetime64("NaT")]})
        >>> df
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Drop the rows where at least one element is null.

        >>> df.dropna()
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Drop the columns where at least one element is null.

        >>> df.dropna(axis='columns')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are null.

        >>> df.dropna(how='all')
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Keep only the rows with at least 2 non-null values.

        >>> df.dropna(thresh=2)
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        2  Catwoman   Bullwhip                 <NA>

        Define in which columns to look for null values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy       born
        0  Alfred  Batmobile 1940-04-25
        """
        if axis == 0:
            result = self._drop_na_rows(how=how, subset=subset, thresh=thresh)
        else:
            result = self._drop_na_columns(
                how=how, subset=subset, thresh=thresh
            )

        return self._mimic_inplace(result, inplace=inplace)

    @_cudf_nvtx_annotate
    def _drop_na_columns(self, how="any", subset=None, thresh=None):
        """
        Drop columns containing nulls
        """
        out_cols = []

        if subset is None:
            df = self
        else:
            df = self.take(subset)

        if thresh is None:
            if how == "all":
                thresh = 1
            else:
                thresh = len(df)

        for name, col in df._data.items():
            try:
                check_col = col.nans_to_nulls()
            except AttributeError:
                check_col = col
            no_threshold_valid_count = (
                len(col) - check_col.null_count
            ) < thresh
            if no_threshold_valid_count:
                continue
            out_cols.append(name)

        return self[out_cols]

    def _drop_na_rows(self, how="any", subset=None, thresh=None):
        """
        Drop null rows from `self`.

        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row.
            any (default) drops rows containing at least
            one null value. all drops only rows containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows.
        thresh : int, optional
            If specified, then drops every row containing
            less than `thresh` non-null values.
        """
        subset = self._preprocess_subset(subset)

        if len(subset) == 0:
            return self.copy(deep=True)

        data_columns = [
            col.nans_to_nulls()
            if isinstance(col, cudf.core.column.NumericalColumn)
            else col
            for col in self._columns
        ]

        return self._from_columns_like_self(
            libcudf.stream_compaction.drop_nulls(
                [*self._index._data.columns, *data_columns],
                how=how,
                keys=self._positions_from_column_names(
                    subset, offset_by_index_columns=True
                ),
                thresh=thresh,
            ),
            self._column_names,
            self._index.names,
        )

    def _apply_boolean_mask(self, boolean_mask: BooleanMask, keep_index=True):
        """Apply boolean mask to each row of `self`.

        Rows corresponding to `False` is dropped.

        If keep_index is False, the index is not preserved.
        """
        if len(boolean_mask.column) != len(self):
            raise IndexError(
                "Boolean mask has wrong length: "
                f"{len(boolean_mask.column)} not {len(self)}"
            )
        return self._from_columns_like_self(
            libcudf.stream_compaction.apply_boolean_mask(
                list(self._index._columns + self._columns)
                if keep_index
                else list(self._columns),
                boolean_mask.column,
            ),
            column_names=self._column_names,
            index_names=self._index.names if keep_index else None,
        )

    def take(self, indices, axis=0):
        """Return a new frame containing the rows specified by *indices*.

        Parameters
        ----------
        indices : array-like
            Array of ints indicating which positions to take.
        axis : Unsupported

        Returns
        -------
        out : Series or DataFrame
            New object with desired subset of rows.

        Examples
        --------
        **Series**
        >>> s = cudf.Series(['a', 'b', 'c', 'd', 'e'])
        >>> s.take([2, 0, 4, 3])
        2    c
        0    a
        4    e
        3    d
        dtype: object

        **DataFrame**

        >>> a = cudf.DataFrame({'a': [1.0, 2.0, 3.0],
        ...                    'b': cudf.Series(['a', 'b', 'c'])})
        >>> a.take([0, 2, 2])
             a  b
        0  1.0  a
        2  3.0  c
        2  3.0  c
        >>> a.take([True, False, True])
             a  b
        0  1.0  a
        2  3.0  c
        """
        if self._get_axis_from_axis_arg(axis) != 0:
            raise NotImplementedError("Only axis=0 is supported.")

        return self._gather(GatherMap(indices, len(self), nullify=False))

    def _reset_index(self, level, drop, col_level=0, col_fill=""):
        """Shared path for DataFrame.reset_index and Series.reset_index."""
        if level is not None:
            if (
                isinstance(level, int)
                and level > 0
                and not isinstance(self.index, MultiIndex)
            ):
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level + 1}"
                )
            if not isinstance(level, (tuple, list)):
                level = (level,)
        _check_duplicate_level_names(level, self._index.names)

        index = self.index._new_index_for_reset_index(level, self.index.name)
        if index is None:
            index = RangeIndex(len(self))
        if drop:
            return self._data, index

        new_column_data = {}
        for name, col in self.index._columns_for_reset_index(level):
            if name == "index" and "index" in self._data:
                name = "level_0"
            name = (
                tuple(
                    name if i == col_level else col_fill
                    for i in range(self._data.nlevels)
                )
                if self._data.multiindex
                else name
            )
            new_column_data[name] = col
        # This is to match pandas where the new data columns are always
        # inserted to the left of existing data columns.
        return (
            ColumnAccessor(
                {**new_column_data, **self._data},
                self._data.multiindex,
                self._data._level_names,
            ),
            index,
        )

    def _first_or_last(
        self, offset, idx: int, op: Callable, side: str, slice_func: Callable
    ) -> "IndexedFrame":
        """Shared code path for ``first`` and ``last``."""
        if not isinstance(self._index, cudf.core.index.DatetimeIndex):
            raise TypeError("'first' only supports a DatetimeIndex index.")
        if not isinstance(offset, str):
            raise NotImplementedError(
                f"Unsupported offset type {type(offset)}."
            )

        if len(self) == 0:
            return self.copy()

        pd_offset = pd.tseries.frequencies.to_offset(offset)
        to_search = op(
            pd.Timestamp(self._index._column.element_indexing(idx)), pd_offset
        )
        if (
            idx == 0
            and not isinstance(pd_offset, pd.tseries.offsets.Tick)
            and pd_offset.is_on_offset(pd.Timestamp(self._index[0]))
        ):
            # Special handle is required when the start time of the index
            # is on the end of the offset. See pandas gh29623 for detail.
            to_search = to_search - pd_offset.base
            return self.loc[:to_search]
        needle = as_column(to_search, dtype=self._index.dtype)
        end_point = int(
            self._index._column.searchsorted(
                needle, side=side
            ).element_indexing(0)
        )
        return slice_func(end_point)

    def first(self, offset):
        """Select initial periods of time series data based on a date offset.

        When having a DataFrame with **sorted** dates as index, this function
        can select the first few rows based on a date offset.

        Parameters
        ----------
        offset: str
            The offset length of the data that will be selected. For instance,
            '1M' will display all rows having their index within the first
            month.

        Returns
        -------
        Series or DataFrame
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a ``DatetimeIndex``

        Examples
        --------
        >>> i = cudf.date_range('2018-04-09', periods=4, freq='2D')
        >>> ts = cudf.DataFrame({'A': [1, 2, 3, 4]}, index=i)
        >>> ts
                    A
        2018-04-09  1
        2018-04-11  2
        2018-04-13  3
        2018-04-15  4
        >>> ts.first('3D')
                    A
        2018-04-09  1
        2018-04-11  2
        """
        # Do not remove until pandas 3.0 support is added.
        assert PANDAS_LT_300, "Need to drop after pandas-3.0 support is added."
        warnings.warn(
            "first is deprecated and will be removed in a future version. "
            "Please create a mask and filter using `.loc` instead",
            FutureWarning,
        )
        return self._first_or_last(
            offset,
            idx=0,
            op=operator.__add__,
            side="left",
            slice_func=lambda i: self.iloc[:i],
        )

    def last(self, offset):
        """Select final periods of time series data based on a date offset.

        When having a DataFrame with **sorted** dates as index, this function
        can select the last few rows based on a date offset.

        Parameters
        ----------
        offset: str
            The offset length of the data that will be selected. For instance,
            '3D' will display all rows having their index within the last 3
            days.

        Returns
        -------
        Series or DataFrame
            A subset of the caller.

        Raises
        ------
        TypeError
            If the index is not a ``DatetimeIndex``

        Examples
        --------
        >>> i = cudf.date_range('2018-04-09', periods=4, freq='2D')
        >>> ts = cudf.DataFrame({'A': [1, 2, 3, 4]}, index=i)
        >>> ts
                    A
        2018-04-09  1
        2018-04-11  2
        2018-04-13  3
        2018-04-15  4
        >>> ts.last('3D')
                    A
        2018-04-13  3
        2018-04-15  4
        """
        # Do not remove until pandas 3.0 support is added.
        assert PANDAS_LT_300, "Need to drop after pandas-3.0 support is added."
        warnings.warn(
            "last is deprecated and will be removed in a future version. "
            "Please create a mask and filter using `.loc` instead",
            FutureWarning,
        )
        return self._first_or_last(
            offset,
            idx=-1,
            op=operator.__sub__,
            side="right",
            slice_func=lambda i: self.iloc[i:],
        )

    @_cudf_nvtx_annotate
    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        ignore_index=False,
    ):
        """Return a random sample of items from an axis of object.

        If reproducible results are required, a random number generator may be
        provided via the `random_state` parameter. This function will always
        produce the same sample given an identical `random_state`.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with `frac`.
            Default = 1 if frac = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with n.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
            `replace == True` is not supported for axis = 1/"columns".
            `replace == False` is not supported for axis = 0/"index" given
            `random_state` is `None` or a cupy random state, and `weights` is
            specified.
        weights : ndarray-like, optional
            Default `None` for uniform probability distribution over rows to
            sample from. If `ndarray` is passed, the length of `weights` should
            equal to the number of rows to sample from, and will be normalized
            to have a sum of 1. Unlike pandas, index alignment is not currently
            not performed.
        random_state : int, numpy/cupy RandomState, or None, default None
            If None, default cupy random state is chosen.
            If int, the seed for the default cupy random state.
            If RandomState, rows-to-sample are generated from the RandomState.
        axis : {0 or `index`, 1 or `columns`, None}, default None
            Axis to sample. Accepts axis number or name.
            Default is stat axis for given data type
            (0 for Series and DataFrames). Series doesn't support axis=1.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing n items
            randomly sampled from the caller object.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":{1, 2, 3, 4, 5}})
        >>> df.sample(3)
           a
        1  2
        3  4
        0  1

        >>> sr = cudf.Series([1, 2, 3, 4, 5])
        >>> sr.sample(10, replace=True)
        1    4
        3    1
        2    4
        0    5
        0    1
        4    5
        4    1
        0    2
        0    3
        3    2
        dtype: int64

        >>> df = cudf.DataFrame(
        ...     {"a": [1, 2], "b": [2, 3], "c": [3, 4], "d": [4, 5]}
        ... )
        >>> df.sample(2, axis=1)
           a  c
        0  1  3
        1  2  4

        .. pandas-compat::
            **DataFrame.sample, Series.sample**

            When sampling from ``axis=0/'index'``, ``random_state`` can be
            either a numpy random state (``numpy.random.RandomState``)
            or a cupy random state (``cupy.random.RandomState``). When a numpy
            random state is used, the output is guaranteed to match the output
            of the corresponding pandas method call, but generating the sample
            maybe slow. If exact pandas equivalence is not required, using a
            cupy random state will achieve better performance,
            especially when sampling large number of
            items. It's advised to use the matching `ndarray` type to
            the random state for the `weights` array.
        """
        axis = 0 if axis is None else self._get_axis_from_axis_arg(axis)
        size = self.shape[axis]

        # Compute `n` from parameter `frac`.
        if frac is None:
            n = 1 if n is None else n
        else:
            if frac > 1 and not replace:
                raise ValueError(
                    "Replace has to be set to `True` when upsampling the "
                    "population `frac` > 1."
                )
            if n is not None:
                raise ValueError(
                    "Please enter a value for `frac` OR `n`, not both."
                )
            n = int(round(size * frac))

        if n > 0 and size == 0:
            raise ValueError(
                "Cannot take a sample larger than 0 when axis is empty."
            )

        if isinstance(random_state, cp.random.RandomState):
            lib = cp
        elif isinstance(random_state, np.random.RandomState):
            lib = np
        else:
            # Construct random state if `random_state` parameter is None or a
            # seed. By default, cupy random state is used to sample rows
            # and numpy is used to sample columns. This is because row data
            # is stored on device, and the column objects are stored on host.
            lib = cp if axis == 0 else np
            random_state = lib.random.RandomState(seed=random_state)

        # Normalize `weights` array.
        if weights is not None:
            if isinstance(weights, str):
                raise NotImplementedError(
                    "Weights specified by string is unsupported yet."
                )

            if size != len(weights):
                raise ValueError(
                    "Weights and axis to be sampled must be of same length."
                )

            weights = lib.asarray(weights)
            weights = weights / weights.sum()

        if axis == 0:
            return self._sample_axis_0(
                n, weights, replace, random_state, ignore_index
            )
        else:
            if isinstance(random_state, cp.random.RandomState):
                raise ValueError(
                    "Sampling from `axis=1`/`columns` with cupy random state"
                    "isn't supported."
                )
            return self._sample_axis_1(
                n, weights, replace, random_state, ignore_index
            )

    def _sample_axis_0(
        self,
        n: int,
        weights: Optional[ColumnLike],
        replace: bool,
        random_state: Union[np.random.RandomState, cp.random.RandomState],
        ignore_index: bool,
    ):
        try:
            gather_map = GatherMap.from_column_unchecked(
                cudf.core.column.as_column(
                    random_state.choice(
                        len(self), size=n, replace=replace, p=weights
                    )
                ),
                len(self),
                nullify=False,
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Random sampling with cupy does not support these inputs."
            ) from e

        return self._gather(gather_map, keep_index=not ignore_index)

    def _sample_axis_1(
        self,
        n: int,
        weights: Optional[ColumnLike],
        replace: bool,
        random_state: np.random.RandomState,
        ignore_index: bool,
    ):
        raise NotImplementedError(
            f"Sampling from axis 1 is not implemented for {self.__class__}."
        )

    def _binaryop(
        self,
        other: Any,
        op: str,
        fill_value: Any = None,
        can_reindex: bool = False,
        *args,
        **kwargs,
    ):
        reflect, op = self._check_reflected_op(op)
        (
            operands,
            out_index,
            can_use_self_column_name,
        ) = self._make_operands_and_index_for_binop(
            other, op, fill_value, reflect, can_reindex
        )
        if operands is NotImplemented:
            return NotImplemented

        level_names = (
            self._data._level_names if can_use_self_column_name else None
        )
        return self._from_data(
            ColumnAccessor(
                type(self)._colwise_binop(operands, op),
                level_names=level_names,
            ),
            index=out_index,
        )

    def _make_operands_and_index_for_binop(
        self,
        other: Any,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
    ) -> Tuple[
        Union[
            Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]],
            NotImplementedType,
        ],
        Optional[cudf.BaseIndex],
        bool,
    ]:
        raise NotImplementedError(
            f"Binary operations are not supported for {self.__class__}"
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ret = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        fname = ufunc.__name__

        if ret is not None:
            return ret

        # Attempt to dispatch all other functions to cupy.
        cupy_func = getattr(cp, fname)
        if cupy_func:
            if ufunc.nin == 2:
                other = inputs[self is inputs[0]]
                inputs, index, _ = self._make_operands_and_index_for_binop(
                    other, fname
                )
            else:
                # This works for Index too
                inputs = {
                    name: (col, None, False, None)
                    for name, col in self._data.items()
                }
                index = self._index

            data = self._apply_cupy_ufunc_to_operands(
                ufunc, cupy_func, inputs, **kwargs
            )

            out = tuple(self._from_data(out, index=index) for out in data)
            return out[0] if ufunc.nout == 1 else out

        return NotImplemented

    @_cudf_nvtx_annotate
    def repeat(self, repeats, axis=None):
        """Repeats elements consecutively.

        Returns a new object of caller type(DataFrame/Series) where each
        element of the current object is repeated consecutively a given
        number of times.

        Parameters
        ----------
        repeats : int, or array of ints
            The number of repetitions for each element. This should
            be a non-negative integer. Repeating 0 times will return
            an empty object.

        Returns
        -------
        Series/DataFrame
            A newly created object of same type as caller
            with repeated elements.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        >>> df
           a   b
        0  1  10
        1  2  20
        2  3  30
        >>> df.repeat(3)
           a   b
        0  1  10
        0  1  10
        0  1  10
        1  2  20
        1  2  20
        1  2  20
        2  3  30
        2  3  30
        2  3  30

        Repeat on Series

        >>> s = cudf.Series([0, 2])
        >>> s
        0    0
        1    2
        dtype: int64
        >>> s.repeat([3, 4])
        0    0
        0    0
        0    0
        1    2
        1    2
        1    2
        1    2
        dtype: int64
        >>> s.repeat(2)
        0    0
        0    0
        1    2
        1    2
        dtype: int64
        """
        res = self._from_columns_like_self(
            Frame._repeat(
                [*self._index._data.columns, *self._columns], repeats, axis
            ),
            self._column_names,
            self._index_names,
        )
        if isinstance(res.index, cudf.DatetimeIndex):
            res.index._freq = None
        return res

    def astype(
        self,
        dtype,
        copy: bool = False,
        errors: Literal["raise", "ignore"] = "raise",
    ):
        """Cast the object to the given dtype.

        Parameters
        ----------
        dtype : data type, or dict of column name -> data type
            Use a :class:`numpy.dtype` or Python type to cast entire DataFrame
            object to the same type. Alternatively, use ``{col: dtype, ...}``,
            where col is a column label and dtype is a :class:`numpy.dtype`
            or Python type to cast one or more of the DataFrame's columns to
            column-specific types.
        copy : bool, default False
            Return a deep-copy when ``copy=True``. Note by default
            ``copy=False`` setting is used and hence changes to
            values then may propagate to other cudf objects.
        errors : {'raise', 'ignore', 'warn'}, default 'raise'
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.

        Returns
        -------
        DataFrame/Series

        Examples
        --------
        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
        >>> df
            a  b
        0  10  1
        1  20  2
        2  30  3
        >>> df.dtypes
        a    int64
        b    int64
        dtype: object

        Cast all columns to `int32`:

        >>> df.astype('int32').dtypes
        a    int32
        b    int32
        dtype: object

        Cast `a` to `float32` using a dictionary:

        >>> df.astype({'a': 'float32'}).dtypes
        a    float32
        b      int64
        dtype: object
        >>> df.astype({'a': 'float32'})
              a  b
        0  10.0  1
        1  20.0  2
        2  30.0  3

        **Series**

        >>> import cudf
        >>> series = cudf.Series([1, 2], dtype='int32')
        >>> series
        0    1
        1    2
        dtype: int32
        >>> series.astype('int64')
        0    1
        1    2
        dtype: int64

        Convert to categorical type:

        >>> series.astype('category')
        0    1
        1    2
        dtype: category
        Categories (2, int64): [1, 2]

        Convert to ordered categorical type with custom ordering:

        >>> cat_dtype = cudf.CategoricalDtype(categories=[2, 1], ordered=True)
        >>> series.astype(cat_dtype)
        0    1
        1    2
        dtype: category
        Categories (2, int64): [2 < 1]

        Note that using ``copy=False`` (enabled by default)
        and changing data on a new Series will
        propagate changes:

        >>> s1 = cudf.Series([1, 2])
        >>> s1
        0    1
        1    2
        dtype: int64
        >>> s2 = s1.astype('int64', copy=False)
        >>> s2[0] = 10
        >>> s1
        0    10
        1     2
        dtype: int64
        """
        if errors not in ("ignore", "raise"):
            raise ValueError("invalid error value specified")

        try:
            data = super().astype(dtype, copy)
        except Exception as e:
            if errors == "raise":
                raise e
            return self

        return self._from_data(data, index=self._index)

    @_cudf_nvtx_annotate
    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by specifying directly index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        index : single label or list-like
            Alternative to specifying axis (``labels, axis=0``
            is equivalent to ``index=labels``).
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        DataFrame or Series
            DataFrame or Series without the removed index or column labels.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis.

        See Also
        --------
        DataFrame.loc : Label-location based indexer for selection by label.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.
        DataFrame.drop_duplicates : Return DataFrame with duplicate rows
            removed, optionally only considering certain columns.
        Series.reindex
            Return only specified index labels of Series
        Series.dropna
            Return series without null values
        Series.drop_duplicates
            Return series with duplicate values removed

        Examples
        --------
        **Series**

        >>> s = cudf.Series([1,2,3], index=['x', 'y', 'z'])
        >>> s
        x    1
        y    2
        z    3
        dtype: int64

        Drop labels x and z

        >>> s.drop(labels=['x', 'z'])
        y    2
        dtype: int64

        Drop a label from the second level in MultiIndex Series.

        >>> midx = cudf.MultiIndex.from_product([[0, 1, 2], ['x', 'y']])
        >>> s = cudf.Series(range(6), index=midx)
        >>> s
        0  x    0
           y    1
        1  x    2
           y    3
        2  x    4
           y    5
        dtype: int64
        >>> s.drop(labels='y', level=1)
        0  x    0
        1  x    2
        2  x    4
        Name: 2, dtype: int64

        **DataFrame**

        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2, 3, 4],
        ...                      "B": [5, 6, 7, 8],
        ...                      "C": [10, 11, 12, 13],
        ...                      "D": [20, 30, 40, 50]})
        >>> df
           A  B   C   D
        0  1  5  10  20
        1  2  6  11  30
        2  3  7  12  40
        3  4  8  13  50

        Drop columns

        >>> df.drop(['B', 'C'], axis=1)
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50
        >>> df.drop(columns=['B', 'C'])
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50

        Drop a row by index

        >>> df.drop([0, 1])
           A  B   C   D
        2  3  7  12  40
        3  4  8  13  50

        Drop columns and/or rows of MultiIndex DataFrame

        >>> midx = cudf.MultiIndex(levels=[['lama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> df = cudf.DataFrame(index=midx, columns=['big', 'small'],
        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...                         [250, 150], [1.5, 0.8], [320, 250],
        ...                         [1, 0.8], [0.3, 0.2]])
        >>> df
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
               length    1.5    1.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
               length    1.5    0.8
        falcon speed   320.0  250.0
               weight    1.0    0.8
               length    0.3    0.2
        >>> df.drop(index='cow', columns='small')
                         big
        lama   speed    45.0
               weight  200.0
               length    1.5
        falcon speed   320.0
               weight    1.0
               length    0.3
        >>> df.drop(index='length', level=1)
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
        falcon speed   320.0  250.0
               weight    1.0    0.8
        """
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and 'index'/'columns'"
                )
            target = labels
        elif index is not None:
            target = index
            axis = 0
        elif columns is not None:
            target = columns
            axis = 1
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', "
                "'index' or 'columns'"
            )

        if inplace:
            out = self
        else:
            out = self.copy()

        if axis in (1, "columns"):
            target = _get_host_unique(target)

            _drop_columns(out, target, errors)
        elif axis in (0, "index"):
            dropped = _drop_rows_by_labels(out, target, level, errors)

            if columns is not None:
                columns = _get_host_unique(columns)
                _drop_columns(dropped, columns, errors)

            out._data = dropped._data
            out._index = dropped._index

        if not inplace:
            return out

    @_cudf_nvtx_annotate
    def _explode(self, explode_column: Any, ignore_index: bool):
        # Helper function for `explode` in `Series` and `Dataframe`, explodes a
        # specified nested column. Other columns' corresponding rows are
        # duplicated. If ignore_index is set, the original index is not
        # exploded and will be replaced with a `RangeIndex`.
        if not isinstance(self._data[explode_column].dtype, ListDtype):
            data = self._data.copy(deep=True)
            idx = None if ignore_index else self._index.copy(deep=True)
            return self.__class__._from_data(data, index=idx)

        column_index = self._column_names.index(explode_column)
        if not ignore_index and self._index is not None:
            index_offset = self._index.nlevels
        else:
            index_offset = 0

        exploded = libcudf.lists.explode_outer(
            [
                *(self._index._data.columns if not ignore_index else ()),
                *self._columns,
            ],
            column_index + index_offset,
        )
        # We must copy inner datatype of the exploded list column to
        # maintain struct dtype key names
        exploded_dtype = cast(
            ListDtype, self._columns[column_index].dtype
        ).element_type
        return self._from_columns_like_self(
            exploded,
            self._column_names,
            self._index_names if not ignore_index else None,
            override_dtypes=(
                exploded_dtype if i == column_index else None
                for i in range(len(self._columns))
            ),
        )

    @_cudf_nvtx_annotate
    def tile(self, count):
        """Repeats the rows `count` times to form a new Frame.

        Parameters
        ----------
        self : input Table containing columns to interleave.
        count : Number of times to tile "rows". Must be non-negative.

        Examples
        --------
        >>> import cudf
        >>> df  = cudf.Dataframe([[8, 4, 7], [5, 2, 3]])
        >>> count = 2
        >>> df.tile(df, count)
           0  1  2
        0  8  4  7
        1  5  2  3
        0  8  4  7
        1  5  2  3

        Returns
        -------
        The indexed frame containing the tiled "rows".
        """
        return self._from_columns_like_self(
            libcudf.reshape.tile(
                [*self._index._columns, *self._columns], count
            ),
            column_names=self._column_names,
            index_names=self._index_names,
        )

    @_cudf_nvtx_annotate
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=no_default,
        group_keys=False,
        squeeze=False,
        observed=True,
        dropna=True,
    ):
        if sort is no_default:
            sort = cudf.get_option("mode.pandas_compatible")

        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if squeeze is not False:
            raise NotImplementedError(
                "squeeze parameter is not yet implemented"
            )

        if not observed:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be specified."
            )
        if group_keys is None:
            group_keys = False

        return (
            self.__class__._resampler(self, by=by)
            if isinstance(by, cudf.Grouper) and by.freq
            else self.__class__._groupby(
                self,
                by=by,
                level=level,
                as_index=as_index,
                dropna=dropna,
                sort=sort,
                group_keys=group_keys,
            )
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Addition",
            op_name="add",
            equivalent_op="frame + other",
            df_op_example=textwrap.dedent(
                """
                >>> df.add(1)
                        angles  degrees
                circle          1      361
                triangle        4      181
                rectangle       5      361
                """,
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.add(b)
                a       2
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.add(b, fill_value=0)
                a       2
                b       1
                c       1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def add(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__add__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Addition",
            op_name="radd",
            equivalent_op="other + frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.radd(1)
                        angles  degrees
                circle          1      361
                triangle        4      181
                rectangle       5      361
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.radd(b)
                a       2
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.radd(b, fill_value=0)
                a       2
                b       1
                c       1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def radd(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__radd__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Subtraction",
            op_name="sub",
            equivalent_op="frame - other",
            df_op_example=textwrap.dedent(
                """
                >>> df.sub(1)
                        angles  degrees
                circle         -1      359
                triangle        2      179
                rectangle       3      359
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.sub(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.sub(b, fill_value=0)
                a       2
                b       1
                c       1
                d      -1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def subtract(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__sub__", fill_value)

    sub = subtract

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Subtraction",
            op_name="rsub",
            equivalent_op="other - frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rsub(1)
                        angles  degrees
                circle          1     -359
                triangle       -2     -179
                rectangle      -3     -359
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rsub(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rsub(b, fill_value=0)
                a       0
                b      -1
                c      -1
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rsub(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rsub__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Multiplication",
            op_name="mul",
            equivalent_op="frame * other",
            df_op_example=textwrap.dedent(
                """
                >>> df.multiply(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.multiply(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.multiply(b, fill_value=0)
                a       1
                b       0
                c       0
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def multiply(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__mul__", fill_value)

    mul = multiply

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Multiplication",
            op_name="rmul",
            equivalent_op="other * frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rmul(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rmul(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rmul(b, fill_value=0)
                a       1
                b       0
                c       0
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rmul(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rmul__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Modulo",
            op_name="mod",
            equivalent_op="frame % other",
            df_op_example=textwrap.dedent(
                """
                >>> df.mod(1)
                        angles  degrees
                circle          0        0
                triangle        0        0
                rectangle       0        0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.mod(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.mod(b, fill_value=0)
                a             0
                b    4294967295
                c    4294967295
                d             0
                e          <NA>
                dtype: int64
                """
            ),
        )
    )
    def mod(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__mod__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Modulo",
            op_name="rmod",
            equivalent_op="other % frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rmod(1)
                            angles  degrees
                circle     4294967295        1
                triangle            1        1
                rectangle           1        1
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rmod(b)
                a       0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rmod(b, fill_value=0)
                a             0
                b             0
                c             0
                d    4294967295
                e          <NA>
                dtype: int64
                """
            ),
        )
    )
    def rmod(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rmod__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Exponential",
            op_name="pow",
            equivalent_op="frame ** other",
            df_op_example=textwrap.dedent(
                """
                >>> df.pow(1)
                        angles  degrees
                circle          0      360
                triangle        2      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.pow(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.pow(b, fill_value=0)
                a       1
                b       1
                c       1
                d       0
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def pow(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__pow__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Exponential",
            op_name="rpow",
            equivalent_op="other ** frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rpow(1)
                        angles  degrees
                circle          1        1
                triangle        1        1
                rectangle       1        1
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rpow(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rpow(b, fill_value=0)
                a       1
                b       0
                c       0
                d       1
                e    <NA>
                dtype: int64
                """
            ),
        )
    )
    def rpow(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rpow__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Integer division",
            op_name="floordiv",
            equivalent_op="frame // other",
            df_op_example=textwrap.dedent(
                """
                >>> df.floordiv(1)
                        angles  degrees
                circle          0      360
                triangle        3      180
                rectangle       4      360
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.floordiv(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.floordiv(b, fill_value=0)
                a                      1
                b    9223372036854775807
                c    9223372036854775807
                d                      0
                e                   <NA>
                dtype: int64
                """
            ),
        )
    )
    def floordiv(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__floordiv__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Integer division",
            op_name="rfloordiv",
            equivalent_op="other // frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rfloordiv(1)
                                        angles  degrees
                circle     9223372036854775807        0
                triangle                     0        0
                rectangle                    0        0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rfloordiv(b)
                a       1
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: int64
                >>> a.rfloordiv(b, fill_value=0)
                a                      1
                b                      0
                c                      0
                d    9223372036854775807
                e                   <NA>
                dtype: int64
                """
            ),
        )
    )
    def rfloordiv(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rfloordiv__", fill_value)

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Floating division",
            op_name="truediv",
            equivalent_op="frame / other",
            df_op_example=textwrap.dedent(
                """
                >>> df.truediv(1)
                        angles  degrees
                circle        0.0    360.0
                triangle      3.0    180.0
                rectangle     4.0    360.0
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.truediv(b)
                a     1.0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: float64
                >>> a.truediv(b, fill_value=0)
                a     1.0
                b     Inf
                c     Inf
                d     0.0
                e    <NA>
                dtype: float64
                """
            ),
        )
    )
    def truediv(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__truediv__", fill_value)

    # Alias for truediv
    div = truediv
    divide = truediv

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Floating division",
            op_name="rtruediv",
            equivalent_op="other / frame",
            df_op_example=textwrap.dedent(
                """
                >>> df.rtruediv(1)
                            angles   degrees
                circle          inf  0.002778
                triangle   0.333333  0.005556
                rectangle  0.250000  0.002778
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.rtruediv(b)
                a     1.0
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: float64
                >>> a.rtruediv(b, fill_value=0)
                a     1.0
                b     0.0
                c     0.0
                d     Inf
                e    <NA>
                dtype: float64
                """
            ),
        )
    )
    def rtruediv(self, other, axis, level=None, fill_value=None):  # noqa: D102
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._binaryop(other, "__rtruediv__", fill_value)

    # Alias for rtruediv
    rdiv = rtruediv

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Equal to",
            op_name="eq",
            equivalent_op="frame == other",
            df_op_example=textwrap.dedent(
                """
                >>> df.eq(1)
                        angles  degrees
                circle      False    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.eq(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.eq(b, fill_value=0)
                a    True
                b   False
                c   False
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def eq(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__eq__", fill_value=fill_value, can_reindex=True
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Not equal to",
            op_name="ne",
            equivalent_op="frame != other",
            df_op_example=textwrap.dedent(
                """
                >>> df.ne(1)
                        angles  degrees
                circle       True     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.ne(b)
                a    False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.ne(b, fill_value=0)
                a   False
                b    True
                c    True
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def ne(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__ne__", fill_value=fill_value, can_reindex=True
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Less than",
            op_name="lt",
            equivalent_op="frame < other",
            df_op_example=textwrap.dedent(
                """
                >>> df.lt(1)
                        angles  degrees
                circle       True    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.lt(b)
                a   False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.lt(b, fill_value=0)
                a   False
                b   False
                c   False
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def lt(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__lt__", fill_value=fill_value, can_reindex=True
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Less than or equal to",
            op_name="le",
            equivalent_op="frame <= other",
            df_op_example=textwrap.dedent(
                """
                >>> df.le(1)
                        angles  degrees
                circle       True    False
                triangle    False    False
                rectangle   False    False
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.le(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.le(b, fill_value=0)
                a    True
                b   False
                c   False
                d    True
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def le(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__le__", fill_value=fill_value, can_reindex=True
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Greater than",
            op_name="gt",
            equivalent_op="frame > other",
            df_op_example=textwrap.dedent(
                """
                >>> df.gt(1)
                        angles  degrees
                circle      False     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.gt(b)
                a   False
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.gt(b, fill_value=0)
                a   False
                b    True
                c    True
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def gt(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__gt__", fill_value=fill_value, can_reindex=True
        )

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        doc_binop_template.format(
            operation="Greater than or equal to",
            op_name="ge",
            equivalent_op="frame >= other",
            df_op_example=textwrap.dedent(
                """
                >>> df.ge(1)
                        angles  degrees
                circle      False     True
                triangle     True     True
                rectangle    True     True
                """
            ),
            ser_op_example=textwrap.dedent(
                """
                >>> a.ge(b)
                a    True
                b    <NA>
                c    <NA>
                d    <NA>
                e    <NA>
                dtype: bool
                >>> a.ge(b, fill_value=0)
                a   True
                b    True
                c    True
                d   False
                e    <NA>
                dtype: bool
                """
            ),
        )
    )
    def ge(self, other, axis="columns", level=None, fill_value=None):  # noqa: D102
        return self._binaryop(
            other=other, op="__ge__", fill_value=fill_value, can_reindex=True
        )

    def _preprocess_subset(self, subset):
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self._data.names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError(f"columns {diff} do not exist")
        return subset

    @_cudf_nvtx_annotate
    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=False,
        na_option="keep",
        ascending=True,
        pct=False,
    ):
        """
        Compute numerical data ranks (1 through n) along axis.

        By default, equal values are assigned a rank that is the average of the
        ranks of those values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Index to direct ranking.
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value
            (i.e. ties):
            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.
        numeric_only : bool, default False
            For DataFrame objects, rank only numeric columns if set to True.
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:
            * keep: assign NaN rank to NaN values
            * top: assign smallest rank to NaN values if ascending
            * bottom: assign highest rank to NaN values if ascending.
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        Returns
        -------
        same type as caller
            Return a Series or DataFrame with data ranks as values.
        """
        if method not in {"average", "min", "max", "first", "dense"}:
            raise KeyError(method)

        method_enum = libcudf.pylibcudf.aggregation.RankMethod[method.upper()]
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                "na_option must be one of 'keep', 'top', or 'bottom'"
            )

        if axis not in (0, "index"):
            raise NotImplementedError(
                f"axis must be `0`/`index`, "
                f"axis={axis} is not yet supported in rank"
            )

        source = self
        if numeric_only:
            if isinstance(
                source, cudf.Series
            ) and not _is_non_decimal_numeric_dtype(self.dtype):
                raise TypeError(
                    "Series.rank does not allow numeric_only=True with "
                    "non-numeric dtype."
                )
            numeric_cols = (
                name
                for name in self._data.names
                if _is_non_decimal_numeric_dtype(self._data[name])
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return source.astype("float64")

        result_columns = libcudf.sort.rank_columns(
            [*source._columns], method_enum, na_option, ascending, pct
        )

        return self.__class__._from_data(
            dict(zip(source._column_names, result_columns)),
            index=source._index,
        ).astype(np.float64)

    def convert_dtypes(
        self,
        infer_objects=True,
        convert_string=True,
        convert_integer=True,
        convert_boolean=True,
        convert_floating=True,
        dtype_backend=None,
    ):
        """
        Convert columns to the best possible nullable dtypes.

        If the dtype is numeric, and consists of all integers, convert
        to an appropriate integer extension type. Otherwise, convert
        to an appropriate floating type.

        All other dtypes are always returned as-is as all dtypes in
        cudf are nullable.
        """
        result = self.copy()

        if convert_floating:
            # cast any floating columns to int64 if
            # they are all integer data:
            for name, col in result._data.items():
                if col.dtype.kind == "f":
                    col = col.fillna(0)
                    if cp.allclose(col, col.astype("int64")):
                        result._data[name] = col.astype("int64")
        return result

    @_warn_no_dask_cudf
    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return [
            type(self),
            str(self._dtypes),
            *[
                normalize_token(cat.categories)
                for cat in self._dtypes.values()
                if cat == "category"
            ],
            normalize_token(self.index),
            normalize_token(self.hash_values().values_host),
        ]


def _check_duplicate_level_names(specified, level_names):
    """Raise if any of `specified` has duplicates in `level_names`."""
    if specified is None:
        return
    if len(set(level_names)) == len(level_names):
        return
    duplicates = {key for key, val in Counter(level_names).items() if val > 1}

    duplicates_specified = [spec for spec in specified if spec in duplicates]
    if not len(duplicates_specified) == 0:
        # Note: pandas raises first encountered duplicates, cuDF raises all.
        raise ValueError(
            f"The names {duplicates_specified} occurs multiple times, use a"
            " level number"
        )


@_cudf_nvtx_annotate
def _get_replacement_values_for_columns(
    to_replace: Any, value: Any, columns_dtype_map: Dict[Any, Any]
) -> Tuple[Dict[Any, bool], Dict[Any, Any], Dict[Any, Any]]:
    """
    Returns a per column mapping for the values to be replaced, new
    values to be replaced with and if all the values are empty.

    Parameters
    ----------
    to_replace : numeric, str, list-like or dict
        Contains the values to be replaced.
    value : numeric, str, list-like, or dict
        Contains the values to replace `to_replace` with.
    columns_dtype_map : dict
        A column to dtype mapping representing dtype of columns.

    Returns
    -------
    all_na_columns : dict
        A dict mapping of all columns if they contain all na values
    to_replace_columns : dict
        A dict mapping of all columns and the existing values that
        have to be replaced.
    values_columns : dict
        A dict mapping of all columns and the corresponding values
        to be replaced with.
    """
    to_replace_columns: Dict[Any, Any] = {}
    values_columns: Dict[Any, Any] = {}
    all_na_columns: Dict[Any, Any] = {}

    if is_scalar(to_replace) and is_scalar(value):
        to_replace_columns = {col: [to_replace] for col in columns_dtype_map}
        values_columns = {col: [value] for col in columns_dtype_map}
    elif cudf.api.types.is_list_like(to_replace) or isinstance(
        to_replace, ColumnBase
    ):
        if is_scalar(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {
                col: [value]
                if _is_non_decimal_numeric_dtype(columns_dtype_map[col])
                else as_column(
                    value,
                    length=len(to_replace),
                    dtype=cudf.dtype(type(value)),
                )
                for col in columns_dtype_map
            }
        elif cudf.api.types.is_list_like(value):
            if len(to_replace) != len(value):
                raise ValueError(
                    f"Replacement lists must be "
                    f"of same length."
                    f" Expected {len(to_replace)}, got {len(value)}."
                )
            else:
                to_replace_columns = {
                    col: to_replace for col in columns_dtype_map
                }
                values_columns = {col: value for col in columns_dtype_map}
        elif cudf.utils.dtypes.is_column_like(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {col: value for col in columns_dtype_map}
        else:
            raise TypeError(
                "value argument must be scalar, list-like or Series"
            )
    elif _is_series(to_replace):
        if value is None or value is no_default:
            to_replace_columns = {
                col: as_column(to_replace.index) for col in columns_dtype_map
            }
            values_columns = {col: to_replace for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in to_replace_columns if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value[col]
                for col in to_replace_columns
                if col in value
            }
        else:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )
    elif is_dict_like(to_replace):
        if value is None or value is no_default:
            to_replace_columns = {
                col: list(to_replace.keys()) for col in columns_dtype_map
            }
            values_columns = {
                col: list(to_replace.values()) for col in columns_dtype_map
            }
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in columns_dtype_map if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value
                for col in columns_dtype_map
                if col in to_replace
            }
        else:
            raise TypeError("value argument must be scalar, dict, or Series")
    else:
        raise TypeError(
            "Expecting 'to_replace' to be either a scalar, array-like, "
            "dict or None, got invalid type "
            f"'{type(to_replace).__name__}'"
        )

    to_replace_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in to_replace_columns.items()
    }
    values_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in values_columns.items()
    }

    for i in to_replace_columns:
        if i in values_columns:
            if isinstance(values_columns[i], list):
                all_na = values_columns[i].count(None) == len(
                    values_columns[i]
                )
            else:
                all_na = False
            all_na_columns[i] = all_na

    return all_na_columns, to_replace_columns, values_columns


def _is_series(obj):
    """
    Checks if the `obj` is of type `cudf.Series`
    instead of checking for isinstance(obj, cudf.Series)
    """
    return isinstance(obj, Frame) and obj.ndim == 1 and obj._index is not None


@_cudf_nvtx_annotate
def _drop_rows_by_labels(
    obj: DataFrameOrSeries,
    labels: Union[ColumnLike, abc.Iterable, str],
    level: Union[int, str],
    errors: str,
) -> DataFrameOrSeries:
    """Remove rows specified by `labels`.

    If `errors="raise"`, an error is raised if some items in `labels` do not
    exist in `obj._index`.

    Will raise if level(int) is greater or equal to index nlevels.
    """
    if isinstance(level, int) and level >= obj.index.nlevels:
        raise ValueError("Param level out of bounds.")

    if not isinstance(labels, cudf.core.single_column_frame.SingleColumnFrame):
        labels = as_column(labels)

    if isinstance(obj.index, cudf.MultiIndex):
        if level is None:
            level = 0

        levels_index = obj.index.get_level_values(level)
        if errors == "raise" and not labels.isin(levels_index).all():
            raise KeyError("One or more values not found in axis")

        if isinstance(level, int):
            ilevel = level
        else:
            ilevel = obj._index.names.index(level)

        # 1. Merge Index df and data df along column axis:
        # | id | ._index df | data column(s) |
        idx_nlv = obj._index.nlevels
        working_df = obj._index.to_frame(index=False)
        working_df.columns = list(range(idx_nlv))
        for i, col in enumerate(obj._data):
            working_df[idx_nlv + i] = obj._data[col]
        # 2. Set `level` as common index:
        # | level | ._index df w/o level | data column(s) |
        working_df = working_df.set_index(level)

        # 3. Use "leftanti" join to drop
        # TODO: use internal API with "leftanti" and specify left and right
        # join keys to bypass logic check
        to_join = cudf.DataFrame(index=cudf.Index(labels, name=level))
        join_res = working_df.join(to_join, how="leftanti")

        # 4. Reconstruct original layout, and rename
        join_res._insert(
            ilevel, name=join_res._index.name, value=join_res._index
        )

        midx = cudf.MultiIndex.from_frame(
            join_res.iloc[:, 0:idx_nlv], names=obj._index.names
        )

        if isinstance(obj, cudf.Series):
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data, index=midx, name=obj.name
            )
        else:
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data,
                index=midx,
                columns=obj._data.to_pandas_index(),
            )

    else:
        if errors == "raise" and not labels.isin(obj.index).all():
            raise KeyError("One or more values not found in axis")

        key_df = cudf.DataFrame._from_data(
            data={},
            index=cudf.Index(
                labels, name=getattr(labels, "name", obj.index.name)
            ),
        )
        if isinstance(obj, cudf.DataFrame):
            res = obj.join(key_df, how="leftanti")
        else:
            res = obj.to_frame(name="tmp").join(key_df, how="leftanti")["tmp"]
            res.name = obj.name
        # Join changes the index to common type,
        # but we need to preserve the type of
        # index being returned, Hence this type-cast.
        res._index = res.index.astype(obj.index.dtype)
        return res


def _is_same_dtype(lhs_dtype, rhs_dtype):
    # Utility specific to `_reindex` to check
    # for matching column dtype.
    if lhs_dtype == rhs_dtype:
        return True
    elif (
        isinstance(lhs_dtype, cudf.CategoricalDtype)
        and isinstance(rhs_dtype, cudf.CategoricalDtype)
        and lhs_dtype.categories.dtype == rhs_dtype.categories.dtype
    ):
        # OK if categories are not all the same
        return True
    elif (
        isinstance(lhs_dtype, cudf.CategoricalDtype)
        and not isinstance(rhs_dtype, cudf.CategoricalDtype)
        and lhs_dtype.categories.dtype == rhs_dtype
    ):
        return True
    elif (
        isinstance(rhs_dtype, cudf.CategoricalDtype)
        and not isinstance(lhs_dtype, cudf.CategoricalDtype)
        and rhs_dtype.categories.dtype == lhs_dtype
    ):
        return True
    else:
        return False
