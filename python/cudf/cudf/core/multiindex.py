# Copyright (c) 2019-2024, NVIDIA CORPORATION.

from __future__ import annotations

import itertools
import numbers
import operator
import pickle
import warnings
from collections import abc
from collections.abc import Generator
from functools import cached_property
from numbers import Integral
from typing import Any, List, MutableMapping, Tuple, Union

import cupy as cp
import numpy as np
import pandas as pd

import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype
from cudf._typing import DataFrameOrSeries
from cudf.api.extensions import no_default
from cudf.api.types import is_integer, is_list_like, is_object_dtype
from cudf.core import column
from cudf.core._base_index import _return_get_indexer_result
from cudf.core.frame import Frame
from cudf.core.index import (
    BaseIndex,
    _get_indexer_basic,
    _lexsorted_equal_range,
    as_index,
)
from cudf.core.join._join_helpers import _match_join_keys
from cudf.utils.dtypes import is_column_like
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import NotIterable, _external_only_api, _is_same_name


def _maybe_indices_to_slice(indices: cp.ndarray) -> Union[slice, cp.ndarray]:
    """Makes best effort to convert an array of indices into a python slice.
    If the conversion is not possible, return input. `indices` are expected
    to be valid.
    """
    # TODO: improve efficiency by avoiding sync.
    if len(indices) == 1:
        x = indices[0].item()
        return slice(x, x + 1)
    if len(indices) == 2:
        x1, x2 = indices[0].item(), indices[1].item()
        return slice(x1, x2 + 1, x2 - x1)
    start, step = indices[0].item(), (indices[1] - indices[0]).item()
    stop = start + step * len(indices)
    if (indices == cp.arange(start, stop, step)).all():
        return slice(start, stop, step)
    return indices


class MultiIndex(Frame, BaseIndex, NotIterable):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes: sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Not yet supported
    names: optional sequence of objects
        Names for each of the index levels.
    copy : bool, default False
        Copy the levels and codes.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.
        Not yet supported

    Attributes
    ----------
    names
    nlevels
    dtypes
    levels
    codes

    Methods
    -------
    from_arrays
    from_tuples
    from_product
    from_frame
    set_levels
    set_codes
    to_frame
    to_flat_index
    sortlevel
    droplevel
    swaplevel
    reorder_levels
    remove_unused_levels
    get_level_values
    get_loc
    drop

    Returns
    -------
    MultiIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.MultiIndex(
    ... levels=[[1, 2], ['blue', 'red']], codes=[[0, 0, 1, 1], [1, 0, 1, 0]])
    MultiIndex([(1,  'red'),
                (1, 'blue'),
                (2,  'red'),
                (2, 'blue')],
               )
    """

    @_cudf_nvtx_annotate
    def __init__(
        self,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,
        **kwargs,
    ):
        if sortorder is not None:
            raise NotImplementedError("sortorder is not yet supported")
        if name is not None:
            raise NotImplementedError(
                "Use `names`, `name` is not yet supported"
            )
        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")
        if not isinstance(codes, cudf.DataFrame) and not isinstance(
            codes[0], (abc.Sequence, np.ndarray, cp.ndarray)
        ):
            raise TypeError("Codes is not a Sequence of sequences")

        if copy:
            if isinstance(codes, cudf.DataFrame):
                codes = codes.copy(deep=True)
            if len(levels) > 0 and isinstance(
                levels[0], (cudf.Index, cudf.Series)
            ):
                levels = [level.copy(deep=True) for level in levels]

        if not isinstance(codes, cudf.DataFrame):
            if len(levels) == len(codes):
                codes = cudf.DataFrame._from_data(
                    {
                        i: column.as_column(code).astype(np.int64)
                        for i, code in enumerate(codes)
                    }
                )
            else:
                raise ValueError(
                    "MultiIndex has unequal number of levels and "
                    "codes and is inconsistent!"
                )

        levels = [cudf.Index(level) for level in levels]

        if len(levels) != len(codes._data):
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )
        if len({c.size for c in codes._data.columns}) != 1:
            raise ValueError(
                "MultiIndex length of codes does not match "
                "and is inconsistent!"
            )

        source_data = {}
        for (column_name, code), level in zip(codes._data.items(), levels):
            if len(code):
                lo, hi = libcudf.reduce.minmax(code)
                if lo.value < -1 or hi.value > len(level) - 1:
                    raise ValueError(
                        f"Codes must be -1 <= codes <= {len(level) - 1}"
                    )
                if lo.value == -1:
                    # Now we can gather and insert null automatically
                    code[code == -1] = np.iinfo(size_type_dtype).min
            result_col = libcudf.copying.gather(
                [level._column], code, nullify=True
            )
            source_data[column_name] = result_col[0]._with_type_metadata(
                level.dtype
            )

        super().__init__(source_data)
        self._levels = levels
        self._codes = codes
        self._name = None
        self.names = names

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def names(self):
        return self._names

    @names.setter  # type: ignore
    @_cudf_nvtx_annotate
    def names(self, value):
        if value is None:
            value = [None] * self.nlevels
        elif not is_list_like(value):
            raise ValueError("Names should be list-like for a MultiIndex")
        elif len(value) != self.nlevels:
            raise ValueError(
                "Length of names must match number of levels in MultiIndex."
            )

        if len(value) == len(set(value)):
            # IMPORTANT: if the provided names are unique,
            # we reconstruct self._data with the names as keys.
            # If they are not unique, the keys of self._data
            # and self._names will be different, which can lead
            # to unexpected behavior in some cases. This is
            # definitely buggy, but we can't disallow non-unique
            # names either...
            self._data = self._data.__class__(
                dict(zip(value, self._data.values())),
                level_names=self._data.level_names,
                verify=False,
            )
        self._names = pd.core.indexes.frozen.FrozenList(value)

    @_cudf_nvtx_annotate
    def to_series(self, index=None, name=None):
        raise NotImplementedError(
            "MultiIndex.to_series isn't implemented yet."
        )

    @_cudf_nvtx_annotate
    def astype(self, dtype, copy: bool = True):
        if not is_object_dtype(dtype):
            raise TypeError(
                "Setting a MultiIndex dtype to anything other than object is "
                "not supported"
            )
        return self

    @_cudf_nvtx_annotate
    def rename(self, names, inplace=False):
        """
        Alter MultiIndex level names

        Parameters
        ----------
        names : list of label
            Names to set, length must be the same as number of levels
        inplace : bool, default False
            If True, modifies objects directly, otherwise returns a new
            ``MultiIndex`` instance

        Returns
        -------
        None or MultiIndex

        Examples
        --------
        Renaming each levels of a MultiIndex to specified name:

        >>> midx = cudf.MultiIndex.from_product(
        ...     [('A', 'B'), (2020, 2021)], names=['c1', 'c2'])
        >>> midx.rename(['lv1', 'lv2'])
        MultiIndex([('A', 2020),
                    ('A', 2021),
                    ('B', 2020),
                    ('B', 2021)],
                names=['lv1', 'lv2'])
        >>> midx.rename(['lv1', 'lv2'], inplace=True)
        >>> midx
        MultiIndex([('A', 2020),
                    ('A', 2021),
                    ('B', 2020),
                    ('B', 2021)],
                names=['lv1', 'lv2'])

        ``names`` argument must be a list, and must have same length as
        ``MultiIndex.levels``:

        >>> midx.rename(['lv0'])
        Traceback (most recent call last):
        ValueError: Length of names must match number of levels in MultiIndex.

        """
        return self.set_names(names, level=None, inplace=inplace)

    @_cudf_nvtx_annotate
    def set_names(self, names, level=None, inplace=False):
        names_is_list_like = is_list_like(names)
        level_is_list_like = is_list_like(level)

        if level is not None and not level_is_list_like and names_is_list_like:
            raise TypeError(
                "Names must be a string when a single level is provided."
            )

        if not names_is_list_like and level is None and self.nlevels > 1:
            raise TypeError("Must pass list-like as `names`.")

        if not names_is_list_like:
            names = [names]
        if level is not None and not level_is_list_like:
            level = [level]

        if level is not None and len(names) != len(level):
            raise ValueError("Length of names must match length of level.")
        if level is None and len(names) != self.nlevels:
            raise ValueError(
                "Length of names must match number of levels in MultiIndex."
            )

        if level is None:
            level = range(self.nlevels)
        else:
            level = [self._level_index_from_level(lev) for lev in level]

        existing_names = list(self.names)
        for i, lev in enumerate(level):
            existing_names[lev] = names[i]
        names = existing_names

        return self._set_names(names=names, inplace=inplace)

    @classmethod
    @_cudf_nvtx_annotate
    def _from_data(
        cls,
        data: MutableMapping,
        name: Any = None,
    ) -> MultiIndex:
        obj = cls.from_frame(cudf.DataFrame._from_data(data=data))
        if name is not None:
            obj.name = name
        return obj

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def name(self):
        return self._name

    @name.setter  # type: ignore
    @_cudf_nvtx_annotate
    def name(self, value):
        self._name = value

    @_cudf_nvtx_annotate
    def copy(
        self,
        names=None,
        deep=False,
        name=None,
    ):
        """Returns copy of MultiIndex object.

        Returns a copy of `MultiIndex`. The `levels` and `codes` value can be
        set to the provided parameters. When they are provided, the returned
        MultiIndex is always newly constructed.

        Parameters
        ----------
        names : sequence of objects, optional (default None)
            Names for each of the index levels.
        deep : Bool (default False)
            If True, `._data`, `._levels`, `._codes` will be copied. Ignored if
            `levels` or `codes` are specified.
        name : object, optional (default None)
            Kept for compatibility with 1-dimensional Index. Should not
            be used.

        Returns
        -------
        Copy of MultiIndex Instance

        Examples
        --------
        >>> df = cudf.DataFrame({'Close': [3400.00, 226.58, 3401.80, 228.91]})
        >>> idx1 = cudf.MultiIndex(
        ... levels=[['2020-08-27', '2020-08-28'], ['AMZN', 'MSFT']],
        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
        ... names=['Date', 'Symbol'])
        >>> idx2 = idx1.copy(
        ... names=['col1', 'col2'])

        >>> df.index = idx1
        >>> df
                             Close
        Date       Symbol
        2020-08-27 AMZN    3400.00
                   MSFT     226.58
        2020-08-28 AMZN    3401.80
                   MSFT     228.91

        >>> df.index = idx2
        >>> df
                           Close
        col1       col2
        2020-08-27 AMZN  3400.00
                   MSFT   226.58
        2020-08-28 AMZN  3401.80
                   MSFT   228.91
        """

        mi = MultiIndex._from_data(self._data.copy(deep=deep))
        if self._levels is not None:
            mi._levels = [idx.copy(deep=deep) for idx in self._levels]
        if self._codes is not None:
            mi._codes = self._codes.copy(deep)
        if names is not None:
            mi.names = names
        elif self.names is not None:
            mi.names = self.names.copy()

        return mi

    @_cudf_nvtx_annotate
    def __repr__(self):
        max_seq_items = pd.get_option("display.max_seq_items") or len(self)

        if len(self) > max_seq_items:
            n = int(max_seq_items / 2) + 1
            # TODO: Update the following two arange calls to
            # a single arange call once arange has support for
            # a vector start/end points.
            indices = column.as_column(range(n))
            indices = indices.append(
                column.as_column(range(len(self) - n, len(self), 1))
            )
            preprocess = self.take(indices)
        else:
            preprocess = self.copy(deep=False)

        if any(col.has_nulls() for col in preprocess._data.columns):
            preprocess_df = preprocess.to_frame(index=False)
            for name, col in preprocess._data.items():
                if isinstance(
                    col,
                    (
                        column.datetime.DatetimeColumn,
                        column.timedelta.TimeDeltaColumn,
                    ),
                ):
                    preprocess_df[name] = col.astype("str").fillna(
                        str(cudf.NaT)
                    )

            tuples_list = list(
                zip(
                    *list(
                        map(lambda val: pd.NA if val is None else val, col)
                        for col in preprocess_df.to_arrow()
                        .to_pydict()
                        .values()
                    )
                )
            )

            preprocess = preprocess.to_pandas(nullable=True)
            preprocess.values[:] = tuples_list
        else:
            preprocess = preprocess.to_pandas(nullable=True)

        output = repr(preprocess)
        output_prefix = self.__class__.__name__ + "("
        output = output.lstrip(output_prefix)
        lines = output.split("\n")

        if len(lines) > 1:
            if "length=" in lines[-1] and len(self) != len(preprocess):
                last_line = lines[-1]
                length_index = last_line.index("length=")
                last_line = last_line[:length_index] + f"length={len(self)})"
                lines = lines[:-1]
                lines.append(last_line)

        data_output = "\n".join(lines)
        return output_prefix + data_output

    @property
    def _codes_frame(self):
        if self._codes is None:
            self._compute_levels_and_codes()
        return self._codes

    @property  # type: ignore
    @_external_only_api("Use ._codes_frame instead")
    @_cudf_nvtx_annotate
    def codes(self):
        """
        Returns the codes of the underlying MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})
        >>> midx = cudf.MultiIndex.from_frame(df)
        >>> midx
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx.codes
        FrozenList([[0, 1, 2], [0, 1, 2]])
        """
        return pd.core.indexes.frozen.FrozenList(
            col.values for col in self._codes_frame._columns
        )

    def get_slice_bound(self, label, side, kind=None):
        raise NotImplementedError()

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def nlevels(self):
        """Integer number of levels in this MultiIndex."""
        return len(self._data)

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def levels(self):
        """
        Returns list of levels in the MultiIndex

        Returns
        -------
        List of Series objects

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 11, 12]})
        >>> cudf.MultiIndex.from_frame(df)
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx = cudf.MultiIndex.from_frame(df)
        >>> midx
        MultiIndex([(1, 10),
                    (2, 11),
                    (3, 12)],
                names=['a', 'b'])
        >>> midx.levels
        [Index([1, 2, 3], dtype='int64', name='a'), Index([10, 11, 12], dtype='int64', name='b')]
        """  # noqa: E501
        if self._levels is None:
            self._compute_levels_and_codes()
        return self._levels

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def ndim(self):
        """Dimension of the data. For MultiIndex ndim is always 2."""
        return 2

    @_cudf_nvtx_annotate
    def _get_level_label(self, level):
        """Get name of the level.

        Parameters
        ----------
        level : int or level name
            if level is name, it will be returned as it is
            else if level is index of the level, then level
            label will be returned as per the index.
        """

        if level in self._data.names:
            return level
        else:
            return self._data.names[level]

    @_cudf_nvtx_annotate
    def isin(self, values, level=None):
        """Return a boolean array where the index values are in values.

        Compute boolean array of whether each index value is found in
        the passed set of values. The length of the returned boolean
        array matches the length of the index.

        Parameters
        ----------
        values : set, list-like, Index or Multi-Index
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index
            is a MultiIndex).

        Returns
        -------
        is_contained : cupy array
            CuPy array of boolean values.

        Notes
        -----
        When `level` is None, `values` can only be MultiIndex, or a
        set/list-like tuples.
        When `level` is provided, `values` can be Index or MultiIndex,
        or a set/list-like tuples.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> midx = cudf.from_pandas(pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color')))
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])
        """
        if level is None:
            if isinstance(values, cudf.MultiIndex):
                values_idx = values
            elif (
                (
                    isinstance(
                        values,
                        (
                            cudf.Series,
                            cudf.Index,
                            cudf.DataFrame,
                            column.ColumnBase,
                        ),
                    )
                )
                or (not is_list_like(values))
                or (
                    is_list_like(values)
                    and len(values) > 0
                    and not isinstance(values[0], tuple)
                )
            ):
                raise TypeError(
                    "values need to be a Multi-Index or set/list-like tuple "
                    "squences  when `level=None`."
                )
            else:
                values_idx = cudf.MultiIndex.from_tuples(
                    values, names=self.names
                )
            self_df = self.to_frame(index=False).reset_index()
            values_df = values_idx.to_frame(index=False)
            idx = self_df.merge(values_df, how="leftsemi")._data["index"]
            res = column.as_column(False, length=len(self))
            res[idx] = True
            result = res.values
        else:
            level_series = self.get_level_values(level)
            result = level_series.isin(values)

        return result

    def where(self, cond, other=None, inplace=False):
        raise NotImplementedError(
            ".where is not supported for MultiIndex operations"
        )

    @_cudf_nvtx_annotate
    def _compute_levels_and_codes(self):
        levels = []

        codes = {}
        for name, col in self._data.items():
            code, cats = cudf.Series._from_data({None: col}).factorize()
            cats.name = name
            codes[name] = code.astype(np.int64)
            levels.append(cats)

        self._levels = levels
        self._codes = cudf.DataFrame._from_data(codes)

    @_cudf_nvtx_annotate
    def _compute_validity_mask(self, index, row_tuple, max_length):
        """Computes the valid set of indices of values in the lookup"""
        lookup = cudf.DataFrame()
        for i, row in enumerate(row_tuple):
            if isinstance(row, slice) and row == slice(None):
                continue
            lookup[i] = cudf.Series(row)
        frame = cudf.DataFrame(dict(enumerate(index._data.columns)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            data_table = cudf.concat(
                [
                    frame,
                    cudf.DataFrame(
                        {
                            "idx": cudf.Series(
                                column.as_column(range(len(frame)))
                            )
                        }
                    ),
                ],
                axis=1,
            )
        # Sort indices in pandas compatible mode
        # because we want the indices to be fetched
        # in a deterministic order.
        # TODO: Remove this after merge/join
        # obtain deterministic ordering.
        if cudf.get_option("mode.pandas_compatible"):
            lookup_order = "_" + "_".join(map(str, lookup._data.names))
            lookup[lookup_order] = column.as_column(range(len(lookup)))
            postprocess = operator.methodcaller(
                "sort_values", by=[lookup_order, "idx"]
            )
        else:
            postprocess = lambda r: r  # noqa: E731
        result = postprocess(lookup.merge(data_table))["idx"]
        # Avoid computing levels unless the result of the merge is empty,
        # which suggests that a KeyError should be raised.
        if len(result) == 0:
            for idx, row in enumerate(row_tuple):
                if row == slice(None):
                    continue
                if row not in index.levels[idx]._column:
                    raise KeyError(row)
        return result

    @_cudf_nvtx_annotate
    def _get_valid_indices_by_tuple(self, index, row_tuple, max_length):
        # Instructions for Slicing
        # if tuple, get first and last elements of tuple
        # if open beginning tuple, get 0 to highest valid_index
        # if open ending tuple, get highest valid_index to len()
        # if not open end or beginning, get range lowest beginning index
        # to highest ending index
        if isinstance(row_tuple, slice):
            if (
                isinstance(row_tuple.start, numbers.Number)
                or isinstance(row_tuple.stop, numbers.Number)
                or row_tuple == slice(None)
            ):
                stop = row_tuple.stop or max_length
                start, stop, step = row_tuple.indices(stop)
                return column.as_column(range(start, stop, step))
            start_values = self._compute_validity_mask(
                index, row_tuple.start, max_length
            )
            stop_values = self._compute_validity_mask(
                index, row_tuple.stop, max_length
            )
            return column.as_column(
                range(start_values.min(), stop_values.max() + 1)
            )
        elif isinstance(row_tuple, numbers.Number):
            return row_tuple
        return self._compute_validity_mask(index, row_tuple, max_length)

    @_cudf_nvtx_annotate
    def _index_and_downcast(self, result, index, index_key):
        if isinstance(index_key, (numbers.Number, slice)):
            index_key = [index_key]
        if (
            len(index_key) > 0 and not isinstance(index_key, tuple)
        ) or isinstance(index_key[0], slice):
            index_key = index_key[0]

        slice_access = isinstance(index_key, slice)
        out_index = cudf.DataFrame()
        # Select the last n-k columns where n is the number of columns and k is
        # the length of the indexing tuple
        size = 0
        if not isinstance(index_key, (numbers.Number, slice)):
            size = len(index_key)
        for k in range(size, len(index._data)):
            out_index.insert(
                out_index._num_columns,
                k,
                cudf.Series._from_data({None: index._data.columns[k]}),
            )

        # determine if we should downcast from a DataFrame to a Series
        need_downcast = (
            isinstance(result, cudf.DataFrame)
            and len(result) == 1  # only downcast if we have a single row
            and not slice_access  # never downcast if we sliced
            and (
                size == 0  # index_key was an integer
                # we indexed into a single row directly, using its label:
                or len(index_key) == self.nlevels
            )
        )
        if need_downcast:
            result = result.T
            return result[result._data.names[0]]

        if len(result) == 0 and not slice_access:
            # Pandas returns an empty Series with a tuple as name
            # the one expected result column
            result = cudf.Series._from_data(
                {}, name=tuple(col[0] for col in index._data.columns)
            )
        elif out_index._num_columns == 1:
            # If there's only one column remaining in the output index, convert
            # it into an Index and name the final index values according
            # to that column's name.
            *_, last_column = index._data.columns
            out_index = as_index(last_column)
            out_index.name = index.names[-1]
            index = out_index
        elif out_index._num_columns > 1:
            # Otherwise pop the leftmost levels, names, and codes from the
            # source index until it has the correct number of columns (n-k)
            result.reset_index(drop=True)
            if index.names is not None:
                result.names = index.names[size:]
            index = MultiIndex(
                levels=index.levels[size:],
                codes=index._codes_frame.iloc[:, size:],
                names=index.names[size:],
            )

        if isinstance(index_key, tuple):
            result.index = index
        return result

    @_cudf_nvtx_annotate
    def _get_row_major(
        self,
        df: DataFrameOrSeries,
        row_tuple: Union[
            numbers.Number, slice, Tuple[Any, ...], List[Tuple[Any, ...]]
        ],
    ) -> DataFrameOrSeries:
        if pd.api.types.is_bool_dtype(
            list(row_tuple) if isinstance(row_tuple, tuple) else row_tuple
        ):
            return df[row_tuple]
        if isinstance(row_tuple, slice):
            if row_tuple.start is None:
                row_tuple = slice(self[0], row_tuple.stop, row_tuple.step)
            if row_tuple.stop is None:
                row_tuple = slice(row_tuple.start, self[-1], row_tuple.step)
        self._validate_indexer(row_tuple)
        valid_indices = self._get_valid_indices_by_tuple(
            df.index, row_tuple, len(df.index)
        )
        indices = cudf.Series(valid_indices)
        result = df.take(indices)
        final = self._index_and_downcast(result, result.index, row_tuple)
        return final

    @_cudf_nvtx_annotate
    def _validate_indexer(
        self,
        indexer: Union[
            numbers.Number, slice, Tuple[Any, ...], List[Tuple[Any, ...]]
        ],
    ):
        if isinstance(indexer, numbers.Number):
            return
        if isinstance(indexer, tuple):
            # drop any slice(None) from the end:
            indexer = tuple(
                itertools.dropwhile(
                    lambda x: x == slice(None), reversed(indexer)
                )
            )[::-1]

            # now check for size
            if len(indexer) > self.nlevels:
                raise IndexError("Indexer size exceeds number of levels")
        elif isinstance(indexer, slice):
            self._validate_indexer(indexer.start)
            self._validate_indexer(indexer.stop)
        else:
            for i in indexer:
                self._validate_indexer(i)

    @_cudf_nvtx_annotate
    def __eq__(self, other):
        if isinstance(other, MultiIndex):
            return np.array(
                [
                    self_col.equals(other_col)
                    for self_col, other_col in zip(
                        self._data.values(), other._data.values()
                    )
                ]
            )
        return NotImplemented

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def size(self):
        # The size of a MultiIndex is only dependent on the number of rows.
        return self._num_rows

    @_cudf_nvtx_annotate
    def take(self, indices):
        if isinstance(indices, cudf.Series) and indices.has_nulls:
            raise ValueError("Column must have no nulls.")
        obj = super().take(indices)
        obj.names = self.names
        return obj

    @_cudf_nvtx_annotate
    def serialize(self):
        header, frames = super().serialize()
        # Overwrite the names in _data with the true names.
        header["column_names"] = pickle.dumps(self.names)
        return header, frames

    @classmethod
    @_cudf_nvtx_annotate
    def deserialize(cls, header, frames):
        # Spoof the column names to construct the frame, then set manually.
        column_names = pickle.loads(header["column_names"])
        header["column_names"] = pickle.dumps(range(0, len(column_names)))
        obj = super().deserialize(header, frames)
        return obj._set_names(column_names)

    @_cudf_nvtx_annotate
    def __getitem__(self, index):
        flatten = isinstance(index, int)

        if isinstance(index, (Integral, abc.Sequence)):
            index = np.array(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            index = column.as_column(range(start, stop, step))
        result = MultiIndex.from_frame(
            self.to_frame(index=False, name=range(0, self.nlevels)).take(
                index
            ),
            names=self.names,
        )

        # we are indexing into a single row of the MultiIndex,
        # return that row as a tuple:
        if flatten:
            return result.to_pandas()[0]

        if self._codes_frame is not None:
            result._codes = self._codes_frame.take(index)
        if self._levels is not None:
            result._levels = self._levels
        return result

    @_cudf_nvtx_annotate
    def to_frame(self, index=True, name=no_default, allow_duplicates=False):
        """
        Create a DataFrame with the levels of the MultiIndex as columns.

        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original MultiIndex.
        name : list / sequence of str, optional
            The passed names should substitute index level names.
        allow_duplicates : bool, optional default False
            Allow duplicate column labels to be created. Note
            that this parameter is non-functional because
            duplicates column labels aren't supported in cudf.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> mi = cudf.MultiIndex.from_tuples([('a', 'c'), ('b', 'd')])
        >>> mi
        MultiIndex([('a', 'c'),
                    ('b', 'd')],
                   )

        >>> df = mi.to_frame()
        >>> df
             0  1
        a c  a  c
        b d  b  d

        >>> df = mi.to_frame(index=False)
        >>> df
           0  1
        0  a  c
        1  b  d

        >>> df = mi.to_frame(name=['x', 'y'])
        >>> df
             x  y
        a c  a  c
        b d  b  d
        """
        # TODO: Currently this function makes a shallow copy, which is
        # incorrect. We want to make a deep copy, otherwise further
        # modifications of the resulting DataFrame will affect the MultiIndex.
        if name is no_default:
            column_names = [
                level if name is None else name
                for level, name in enumerate(self.names)
            ]
        else:
            if not is_list_like(name):
                raise TypeError(
                    "'name' must be a list / sequence of column names."
                )
            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have the same length as "
                    "number of levels on index."
                )
            column_names = name

        all_none_names = None
        if not (
            all_none_names := all(x is None for x in column_names)
        ) and len(column_names) != len(set(column_names)):
            raise ValueError("Duplicate column names are not allowed")
        df = cudf.DataFrame._from_data(
            data=self._data,
            columns=column_names
            if name is not no_default and not all_none_names
            else None,
        )

        if index:
            df = df.set_index(self)

        return df

    @_cudf_nvtx_annotate
    def get_level_values(self, level):
        """
        Return the values at the requested level

        Parameters
        ----------
        level : int or label

        Returns
        -------
        An Index containing the values at the requested level.
        """
        colnames = self._data.names
        if level not in colnames:
            if isinstance(level, int):
                if level < 0:
                    level = level + len(colnames)
                if level < 0 or level >= len(colnames):
                    raise IndexError(f"Invalid level number: '{level}'")
                level_idx = level
                level = colnames[level_idx]
            elif level in self.names:
                level_idx = list(self.names).index(level)
                level = colnames[level_idx]
            else:
                raise KeyError(f"Level not found: '{level}'")
        else:
            level_idx = colnames.index(level)
        level_values = as_index(self._data[level], name=self.names[level_idx])
        return level_values

    def _is_numeric(self):
        return False

    def _is_boolean(self):
        return False

    def _is_integer(self):
        return False

    def _is_floating(self):
        return False

    def _is_object(self):
        return False

    def _is_categorical(self):
        return False

    def _is_interval(self):
        return False

    @classmethod
    @_cudf_nvtx_annotate
    def _concat(cls, objs):
        source_data = [o.to_frame(index=False) for o in objs]

        # TODO: Verify if this is really necessary or if we can rely on
        # DataFrame._concat.
        if len(source_data) > 1:
            colnames = source_data[0]._data.to_pandas_index()
            for obj in source_data[1:]:
                obj.columns = colnames

        source_data = cudf.DataFrame._concat(source_data)
        try:
            # Only set names if all objs have the same names
            (names,) = {o.names for o in objs} - {None}
        except ValueError:
            names = [None] * source_data._num_columns
        return cudf.MultiIndex.from_frame(source_data, names=names)

    @classmethod
    @_cudf_nvtx_annotate
    def from_tuples(cls, tuples, names=None):
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> cudf.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_tuples(tuples, names=names)
        return cls.from_pandas(pdi)

    @_cudf_nvtx_annotate
    def to_numpy(self):
        return self.values_host

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def values_host(self):
        """
        Return a numpy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out : numpy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.values_host
        array([(1, 1), (1, 5), (3, 2), (4, 2), (5, 1)], dtype=object)
        >>> type(midx.values_host)
        <class 'numpy.ndarray'>
        """
        return self.to_pandas().values

    @property  # type: ignore
    @_cudf_nvtx_annotate
    def values(self):
        """
        Return a CuPy representation of the MultiIndex.

        Only the values in the MultiIndex will be returned.

        Returns
        -------
        out: cupy.ndarray
            The values of the MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx.values
        array([[1, 1],
            [1, 5],
            [3, 2],
            [4, 2],
            [5, 1]])
        >>> type(midx.values)
        <class 'cupy...ndarray'>
        """
        if cudf.get_option("mode.pandas_compatible"):
            raise NotImplementedError(
                "Unable to create a cupy array with tuples."
            )
        return self.to_frame(index=False).values

    @classmethod
    @_cudf_nvtx_annotate
    def from_frame(cls, df, names=None):
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip
        >>> cudf.MultiIndex.from_frame(df)
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> cudf.MultiIndex.from_frame(df, names=['state', 'observation'])
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        obj = cls.__new__(cls)
        super(cls, obj).__init__()

        source_data = df.copy(deep=False)
        source_data.reset_index(drop=True, inplace=True)
        if isinstance(source_data, pd.DataFrame):
            source_data = cudf.DataFrame.from_pandas(source_data)

        names = names if names is not None else source_data._data.names
        # if names are unique
        # try using those as the source_data column names:
        if len(dict.fromkeys(names)) == len(names):
            source_data.columns = names
        obj._name = None
        obj._data = source_data._data
        obj.names = names
        obj._codes = None
        obj._levels = None
        return obj

    @classmethod
    @_cudf_nvtx_annotate
    def from_product(cls, arrays, names=None):
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        names : list / sequence of str, optional
            Names for the levels in the index.
            If not explicitly provided, names will be inferred from the
            elements of iterables if an element has a name attribute

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> cudf.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_product(arrays, names=names)
        return cls.from_pandas(pdi)

    @classmethod
    @_cudf_nvtx_annotate
    def from_arrays(
        cls,
        arrays,
        sortorder=None,
        names=None,
    ) -> MultiIndex:
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays : list / sequence of array-likes
            Each array-like gives one level's value for each data point.
            len(arrays) is the number of levels.
        sortorder : optional int
            Not yet supported
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        MultiIndex

        See Also
        --------
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.
        MultiIndex.from_frame : Make a MultiIndex from a DataFrame.

        Examples
        --------
        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> cudf.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        # Imported here due to circular import
        from cudf.core.algorithms import factorize

        error_msg = "Input must be a list / sequence of array-likes."
        if not is_list_like(arrays):
            raise TypeError(error_msg)
        codes = []
        levels = []
        for array in arrays:
            if not (is_list_like(array) or is_column_like(array)):
                raise TypeError(error_msg)
            code, level = factorize(array, sort=True)
            codes.append(code)
            levels.append(level)
        return cls(
            codes=codes, levels=levels, sortorder=sortorder, names=names
        )

    @_cudf_nvtx_annotate
    def _poplevels(self, level):
        """
        Remove and return the specified levels from self.

        Parameters
        ----------
        level : level name or index, list
            One or more levels to remove

        Returns
        -------
        Index composed of the removed levels. If only a single level
        is removed, a flat index is returned. If no levels are specified
        (empty list), None is returned.
        """
        if not pd.api.types.is_list_like(level):
            level = (level,)

        ilevels = sorted(self._level_index_from_level(lev) for lev in level)

        if not ilevels:
            return None

        popped_data = {}
        popped_names = []
        names = list(self.names)

        # build the popped data and names
        for i in ilevels:
            n = self._data.names[i]
            popped_data[n] = self._data[n]
            popped_names.append(self.names[i])

        # pop the levels out from self
        # this must be done iterating backwards
        for i in reversed(ilevels):
            n = self._data.names[i]
            names.pop(i)
            popped_data[n] = self._data.pop(n)

        # construct the popped result
        popped = cudf.core.index._index_from_data(popped_data)
        popped.names = popped_names

        # update self
        self.names = names
        self._compute_levels_and_codes()

        return popped

    @_cudf_nvtx_annotate
    def swaplevel(self, i=-2, j=-1):
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        Examples
        --------
        >>> import cudf
        >>> mi = cudf.MultiIndex(levels=[['a', 'b'], ['bb', 'aa']],
        ...                    codes=[[0, 0, 1, 1], [0, 1, 0, 1]])
        >>> mi
        MultiIndex([('a', 'bb'),
            ('a', 'aa'),
            ('b', 'bb'),
            ('b', 'aa')],
           )
        >>> mi.swaplevel(0, 1)
        MultiIndex([('bb', 'a'),
            ('aa', 'a'),
            ('bb', 'b'),
            ('aa', 'b')],
           )
        """
        name_i = self._data.names[i] if isinstance(i, int) else i
        name_j = self._data.names[j] if isinstance(j, int) else j
        new_data = {}
        for k, v in self._data.items():
            if k not in (name_i, name_j):
                new_data[k] = v
            elif k == name_i:
                new_data[name_j] = self._data[name_j]
            elif k == name_j:
                new_data[name_i] = self._data[name_i]
        midx = MultiIndex._from_data(new_data)
        if all(n is None for n in self.names):
            midx = midx.set_names(self.names)
        return midx

    @_cudf_nvtx_annotate
    def droplevel(self, level=-1):
        """
        Removes the specified levels from the MultiIndex.

        Parameters
        ----------
        level : level name or index, list-like
            Integer, name or list of such, specifying one or more
            levels to drop from the MultiIndex

        Returns
        -------
        A MultiIndex or Index object, depending on the number of remaining
        levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.MultiIndex.from_frame(
        ...     cudf.DataFrame(
        ...         {
        ...             "first": ["a", "a", "a", "b", "b", "b"],
        ...             "second": [1, 1, 2, 2, 3, 3],
        ...             "third": [0, 1, 2, 0, 1, 2],
        ...         }
        ...     )
        ... )

        Dropping level by index:

        >>> idx.droplevel(0)
        MultiIndex([(1, 0),
                    (1, 1),
                    (2, 2),
                    (2, 0),
                    (3, 1),
                    (3, 2)],
                   names=['second', 'third'])

        Dropping level by name:

        >>> idx.droplevel("first")
        MultiIndex([(1, 0),
                    (1, 1),
                    (2, 2),
                    (2, 0),
                    (3, 1),
                    (3, 2)],
                   names=['second', 'third'])

        Dropping multiple levels:

        >>> idx.droplevel(["first", "second"])
        Index([0, 1, 2, 0, 1, 2], dtype='int64', name='third')
        """
        mi = self.copy(deep=False)
        mi._poplevels(level)
        if mi.nlevels == 1:
            return mi.get_level_values(mi.names[0])
        else:
            return mi

    @_cudf_nvtx_annotate
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.MultiIndex:
        # cudf uses np.iinfo(size_type_dtype).min as missing code
        # pandas uses -1 as missing code
        pd_codes = self._codes_frame.replace(np.iinfo(size_type_dtype).min, -1)
        return pd.MultiIndex(
            levels=[
                level.to_pandas(nullable=nullable, arrow_type=arrow_type)
                for level in self.levels
            ],
            codes=[col.values_host for col in pd_codes._columns],
            names=self.names,
        )

    @classmethod
    @_cudf_nvtx_annotate
    def from_pandas(cls, multiindex: pd.MultiIndex, nan_as_null=no_default):
        """
        Convert from a Pandas MultiIndex

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> pmi = pd.MultiIndex(levels=[['a', 'b'], ['c', 'd']],
        ...                     codes=[[0, 1], [1, 1]])
        >>> cudf.from_pandas(pmi)
        MultiIndex([('a', 'd'),
                    ('b', 'd')],
                   )
        """
        if not isinstance(multiindex, pd.MultiIndex):
            raise TypeError("not a pandas.MultiIndex")
        if nan_as_null is no_default:
            nan_as_null = (
                False if cudf.get_option("mode.pandas_compatible") else None
            )
        levels = [
            cudf.Index.from_pandas(level, nan_as_null=nan_as_null)
            for level in multiindex.levels
        ]
        return cls(
            levels=levels, codes=multiindex.codes, names=multiindex.names
        )

    @cached_property  # type: ignore
    @_cudf_nvtx_annotate
    def is_unique(self):
        return len(self) == len(self.unique())

    @property
    def dtype(self):
        return np.dtype("O")

    @cached_property  # type: ignore
    @_cudf_nvtx_annotate
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._is_sorted(ascending=None, null_position=None)

    @cached_property  # type: ignore
    @_cudf_nvtx_annotate
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        return self._is_sorted(
            ascending=[False] * len(self.levels), null_position=None
        )

    @_cudf_nvtx_annotate
    def fillna(self, value):
        """
        Fill null values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill nulls. This value cannot be a
            list-likes.

        Returns
        -------
        filled : MultiIndex

        Examples
        --------
        >>> import cudf
        >>> index = cudf.MultiIndex(
        ...         levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...       )
        >>> index
        MultiIndex([( 'a',  '1'),
                    ( 'a',  '5'),
                    ( 'b', <NA>),
                    ( 'c', <NA>),
                    (<NA>,  '1')],
                   names=['x', 'y'])
        >>> index.fillna('hello')
        MultiIndex([(    'a',     '1'),
                    (    'a',     '5'),
                    (    'b', 'hello'),
                    (    'c', 'hello'),
                    ('hello',     '1')],
                   names=['x', 'y'])
        """

        return super().fillna(value=value)

    @_cudf_nvtx_annotate
    def unique(self):
        return self.drop_duplicates(keep="first")

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in MultiIndex object
        to `<NA>` as a preprocessing step to `__repr__` methods.
        """
        index_df = self.to_frame(index=False, name=list(range(self.nlevels)))
        return MultiIndex.from_frame(
            index_df._clean_nulls_from_dataframe(index_df), names=self.names
        )

    @_cudf_nvtx_annotate
    def memory_usage(self, deep=False):
        usage = sum(col.memory_usage for col in self._data.columns)
        if self.levels:
            for level in self.levels:
                usage += level.memory_usage(deep=deep)
        if self._codes_frame:
            for col in self._codes_frame._data.columns:
                usage += col.memory_usage
        return usage

    @_cudf_nvtx_annotate
    def difference(self, other, sort=None):
        if hasattr(other, "to_pandas"):
            other = other.to_pandas()
        return cudf.from_pandas(self.to_pandas().difference(other, sort))

    @_cudf_nvtx_annotate
    def append(self, other):
        """
        Append a collection of MultiIndex objects together

        Parameters
        ----------
        other : MultiIndex or list/tuple of MultiIndex objects

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.MultiIndex(
        ...     levels=[[1, 2], ['blue', 'red']],
        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ... )
        >>> idx2 = cudf.MultiIndex(
        ...     levels=[[3, 4], ['blue', 'red']],
        ...     codes=[[0, 0, 1, 1], [1, 0, 1, 0]]
        ... )
        >>> idx1
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   )
        >>> idx2
        MultiIndex([(3,  'red'),
                    (3, 'blue'),
                    (4,  'red'),
                    (4, 'blue')],
                   )
        >>> idx1.append(idx2)
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue'),
                    (3,  'red'),
                    (3, 'blue'),
                    (4,  'red'),
                    (4, 'blue')],
                   )
        """
        if isinstance(other, (list, tuple)):
            to_concat = [self]
            to_concat.extend(other)
        else:
            to_concat = [self, other]

        for obj in to_concat:
            if not isinstance(obj, MultiIndex):
                raise TypeError(
                    f"all objects should be of type "
                    f"MultiIndex for MultiIndex.append, "
                    f"found object of type: {type(obj)}"
                )

        return MultiIndex._concat(to_concat)

    @_cudf_nvtx_annotate
    def __array_function__(self, func, types, args, kwargs):
        cudf_df_module = MultiIndex

        for submodule in func.__module__.split(".")[1:]:
            # point cudf to the correct submodule
            if hasattr(cudf_df_module, submodule):
                cudf_df_module = getattr(cudf_df_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [cudf_df_module, np.ndarray]

        for t in types:
            if t not in handled_types:
                return NotImplemented

        if hasattr(cudf_df_module, fname):
            cudf_func = getattr(cudf_df_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    def _level_index_from_level(self, level):
        """
        Return level index from given level name or index
        """
        try:
            return self.names.index(level)
        except ValueError:
            if not is_integer(level):
                raise KeyError(f"Level {level} not found")
            if level < 0:
                level += self.nlevels
            if level >= self.nlevels:
                raise IndexError(
                    f"Level {level} out of bounds. "
                    f"Index has {self.nlevels} levels."
                ) from None
            return level

    @_cudf_nvtx_annotate
    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        if tolerance is not None:
            raise NotImplementedError(
                "Parameter tolerance is not supported yet."
            )
        if method == "nearest":
            raise NotImplementedError(
                f"{method=} is not supported yet for MultiIndex."
            )
        if method in {"ffill", "bfill", "pad", "backfill"} and not (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        ):
            raise ValueError(
                "index must be monotonic increasing or decreasing"
            )

        result = column.as_column(
            -1,
            length=len(target),
            dtype=libcudf.types.size_type_dtype,
        )
        if not len(self):
            return _return_get_indexer_result(result.values)
        try:
            target = cudf.MultiIndex.from_tuples(target)
        except TypeError:
            return _return_get_indexer_result(result.values)

        join_keys = [
            _match_join_keys(lcol, rcol, "inner")
            for lcol, rcol in zip(target._data.columns, self._data.columns)
        ]
        join_keys = map(list, zip(*join_keys))
        scatter_map, indices = libcudf.join.join(
            *join_keys,
            how="inner",
        )
        (result,) = libcudf.copying.scatter([indices], scatter_map, [result])
        result_series = cudf.Series(result)

        if method in {"ffill", "bfill", "pad", "backfill"}:
            result_series = _get_indexer_basic(
                index=self,
                positions=result_series,
                method=method,
                target_col=target.to_frame(index=False)[
                    list(range(0, self.nlevels))
                ],
                tolerance=tolerance,
            )
        elif method is not None:
            raise ValueError(
                f"{method=} is unsupported, only supported values are: "
                "{['ffill'/'pad', 'bfill'/'backfill', None]}"
            )

        return _return_get_indexer_result(result_series.to_cupy())

    @_cudf_nvtx_annotate
    def get_loc(self, key):
        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )
        is_unique = self.is_unique
        key = (key,) if not isinstance(key, tuple) else key

        # Handle partial key search. If length of `key` is less than `nlevels`,
        # Only search levels up to `len(key)` level.
        key_as_table = cudf.core.frame.Frame(
            {i: column.as_column(k, length=1) for i, k in enumerate(key)}
        )
        partial_index = self.__class__._from_data(
            data=self._data.select_by_index(slice(key_as_table._num_columns))
        )
        (
            lower_bound,
            upper_bound,
            sort_inds,
        ) = _lexsorted_equal_range(partial_index, key_as_table, is_sorted)

        if lower_bound == upper_bound:
            raise KeyError(key)

        if is_unique and lower_bound + 1 == upper_bound:
            # Indices are unique (Pandas constraint), search result is unique,
            # return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        true_inds = sort_inds.slice(lower_bound, upper_bound).values
        true_inds = _maybe_indices_to_slice(true_inds)
        if isinstance(true_inds, slice):
            return true_inds

        # Not sorted and not unique. Return a boolean mask
        mask = cp.full(self._data.nrows, False)
        mask[true_inds] = True
        return mask

    def _get_reconciled_name_object(self, other) -> MultiIndex:
        """
        If the result of a set operation will be self,
        return self, unless the names change, in which
        case make a shallow copy of self.
        """
        names = self._maybe_match_names(other)
        if self.names != names:
            return self.rename(names)
        return self

    def _maybe_match_names(self, other):
        """
        Try to find common names to attach to the result of an operation
        between a and b. Return a consensus list of names if they match
        at least partly or list of None if they have completely
        different names.
        """
        if len(self.names) != len(other.names):
            return [None] * len(self.names)
        return [
            self_name if _is_same_name(self_name, other_name) else None
            for self_name, other_name in zip(self.names, other.names)
        ]

    @_cudf_nvtx_annotate
    def union(self, other, sort=None):
        if not isinstance(other, MultiIndex):
            msg = "other must be a MultiIndex or a list of tuples"
            try:
                other = MultiIndex.from_tuples(other, names=self.names)
            except (ValueError, TypeError) as err:
                # ValueError raised by tuples_to_object_array if we
                #  have non-object dtype
                raise TypeError(msg) from err

        if sort not in {None, False}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"None or False; {sort} was passed."
            )

        if not len(other) or self.equals(other):
            return self._get_reconciled_name_object(other)
        elif not len(self):
            return other._get_reconciled_name_object(self)

        return self._union(other, sort=sort)

    @_cudf_nvtx_annotate
    def _union(self, other, sort=None):
        # TODO: When to_frame is refactored to return a
        # deep copy in future, we should push most of the common
        # logic between MultiIndex._union & BaseIndex._union into
        # Index._union.
        other_df = other.copy(deep=True).to_frame(index=False)
        self_df = self.copy(deep=True).to_frame(index=False)
        col_names = list(range(0, self.nlevels))
        self_df.columns = col_names
        other_df.columns = col_names
        self_df["order"] = self_df.index
        other_df["order"] = other_df.index

        result_df = self_df.merge(other_df, on=col_names, how="outer")
        result_df = result_df.sort_values(
            by=result_df._data.to_pandas_index()[self.nlevels :],
            ignore_index=True,
        )

        midx = MultiIndex.from_frame(result_df.iloc[:, : self.nlevels])
        midx.names = self.names if self.names == other.names else None
        if sort in {None, True} and len(other):
            return midx.sort_values()
        return midx

    @_cudf_nvtx_annotate
    def _intersection(self, other, sort=None):
        if self.names != other.names:
            deep = True
            col_names = list(range(0, self.nlevels))
            res_name = (None,) * self.nlevels
        else:
            deep = False
            col_names = None
            res_name = self.names

        other_df = other.copy(deep=deep).to_frame(index=False)
        self_df = self.copy(deep=deep).to_frame(index=False)
        if col_names is not None:
            other_df.columns = col_names
            self_df.columns = col_names

        result_df = cudf.merge(self_df, other_df, how="inner")
        midx = self.__class__.from_frame(result_df, names=res_name)
        if sort in {None, True} and len(other):
            return midx.sort_values()
        return midx

    @_cudf_nvtx_annotate
    def _copy_type_metadata(
        self: MultiIndex, other: MultiIndex, *, override_dtypes=None
    ) -> MultiIndex:
        res = super()._copy_type_metadata(other)
        if isinstance(other, MultiIndex):
            res._names = other._names
        return res

    @_cudf_nvtx_annotate
    def _split_columns_by_levels(
        self, levels: tuple, *, in_levels: bool
    ) -> Generator[tuple[Any, column.ColumnBase], None, None]:
        # This function assumes that for levels with duplicate names, they are
        # specified by indices, not name by ``levels``. E.g. [None, None] can
        # only be specified by 0, 1, not "None".
        level_names = list(self.names)
        level_indices = {
            lv if isinstance(lv, int) else level_names.index(lv)
            for lv in levels
        }
        for i, (name, col) in enumerate(zip(self.names, self._data.columns)):
            if in_levels and i in level_indices:
                name = f"level_{i}" if name is None else name
                yield name, col
            elif not in_levels and i not in level_indices:
                yield name, col

    @_cudf_nvtx_annotate
    def _new_index_for_reset_index(
        self, levels: tuple | None, name
    ) -> None | BaseIndex:
        """Return the new index after .reset_index"""
        if levels is None:
            return None

        index_columns, index_names = [], []
        for name, col in self._split_columns_by_levels(
            levels, in_levels=False
        ):
            index_columns.append(col)
            index_names.append(name)

        if not index_columns:
            # None is caught later to return RangeIndex
            return None

        index = cudf.core.index._index_from_data(
            dict(enumerate(index_columns)),
            name=name,
        )
        if isinstance(index, type(self)):
            index.names = index_names
        else:
            index.name = index_names[0]
        return index

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, column.ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        if levels is None:
            for i, (col, name) in enumerate(
                zip(self._data.columns, self.names)
            ):
                yield f"level_{i}" if name is None else name, col
        else:
            yield from self._split_columns_by_levels(levels, in_levels=True)

    def repeat(self, repeats, axis=None):
        return self._from_data(
            self._data._from_columns_like_self(
                super()._repeat([*self._columns], repeats, axis)
            )
        )
