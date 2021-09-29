# Copyright (c) 2019-2021, NVIDIA CORPORATION.

from __future__ import annotations

import itertools
import numbers
import pickle
import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, List, MutableMapping, Optional, Tuple, Union

import cupy
import numpy as np
import pandas as pd
from pandas._config import get_option

import cudf
from cudf import _lib as libcudf
from cudf._typing import DataFrameOrSeries
from cudf.api.types import is_integer, is_list_like
from cudf.core import column
from cudf.core._compat import PANDAS_GE_120
from cudf.core.frame import Frame
from cudf.core.index import BaseIndex, _lexsorted_equal_range, as_index
from cudf.utils.utils import _maybe_indices_to_slice, cached_property


class MultiIndex(Frame, BaseIndex):
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
            codes[0], (Sequence, np.ndarray)
        ):
            raise TypeError("Codes is not a Sequence of sequences")

        if copy:
            if isinstance(codes, cudf.DataFrame):
                codes = codes.copy(deep=True)
            if len(levels) > 0 and isinstance(levels[0], cudf.Series):
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

        levels = [cudf.Series(level) for level in levels]

        if len(levels) != len(codes.columns):
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )
        if len(set(c.size for c in codes._data.columns)) != 1:
            raise ValueError(
                "MultiIndex length of codes does not match "
                "and is inconsistent!"
            )
        for level, code in zip(levels, codes._data.columns):
            if code.max() > len(level) - 1:
                raise ValueError(
                    "MultiIndex code %d contains value %d larger "
                    "than maximum level size at this position"
                )

        source_data = {}
        for i, (column_name, col) in enumerate(codes._data.items()):
            if -1 in col.values:
                level = cudf.DataFrame(
                    {column_name: [None] + list(levels[i])},
                    index=range(-1, len(levels[i])),
                )
            else:
                level = cudf.DataFrame({column_name: levels[i]})

            source_data[column_name] = libcudf.copying.gather(level, col)[0][
                column_name
            ]

        super().__init__(source_data)
        self._levels = levels
        self._codes = codes
        self._name = None
        self.names = names

    @property
    def names(self):
        return self._names

    @names.setter
    def names(self, value):
        value = [None] * self.nlevels if value is None else value

        if len(value) == len(set(value)):
            # IMPORTANT: if the provided names are unique,
            # we reconstruct self._data with the names as keys.
            # If they are not unique, the keys of self._data
            # and self._names will be different, which can lead
            # to unexpected behaviour in some cases. This is
            # definitely buggy, but we can't disallow non-unique
            # names either...
            self._data = self._data.__class__._create_unsafe(
                dict(zip(value, self._data.values())),
                level_names=self._data.level_names,
            )
        self._names = pd.core.indexes.frozen.FrozenList(value)

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
        --------
        None or MultiIndex

        Examples
        --------
        Renaming each levels of a MultiIndex to specified name:

        >>> midx = cudf.MultiIndex.from_product(
                [('A', 'B'), (2020, 2021)], names=['c1', 'c2'])
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
        for i, l in enumerate(level):
            existing_names[l] = names[i]
        names = existing_names

        return self._set_names(names=names, inplace=inplace)

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[cudf.core.index.BaseIndex] = None,
        name: Any = None,
    ) -> MultiIndex:
        assert index is None
        obj = cls.from_frame(cudf.DataFrame._from_data(data))
        if name is not None:
            obj.name = name
        return obj

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def copy(
        self,
        names=None,
        dtype=None,
        levels=None,
        codes=None,
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
        dtype : object, optional (default None)
            MultiIndex dtype, only supports None or object type
        levels : sequence of arrays, optional (default None)
            The unique labels for each level. Original values used if None.
        codes : sequence of arrays, optional (default None)
            Integers for each level designating which label at each location.
            Original values used if None.
        deep : Bool (default False)
            If True, `._data`, `._levels`, `._codes` will be copied. Ignored if
            `levels` or `codes` are specified.
        name : object, optional (defulat None)
            To keep consistent with `Index.copy`, should not be used.

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
        ... levels=[['day1', 'day2'], ['com1', 'com2']],
        ... codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
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
        col1 col2
        day1 com1  3400.00
             com2   226.58
        day2 com1  3401.80
             com2   228.91

        """

        dtype = object if dtype is None else dtype
        if not pd.core.dtypes.common.is_object_dtype(dtype):
            raise TypeError("Dtype for MultiIndex only supports object type.")

        # ._data needs to be rebuilt
        if levels is not None or codes is not None:
            if self._levels is None or self._codes is None:
                self._compute_levels_and_codes()
            levels = self._levels if levels is None else levels
            codes = self._codes if codes is None else codes
            names = self.names if names is None else names

            mi = MultiIndex(levels=levels, codes=codes, names=names, copy=deep)
            return mi

        mi = MultiIndex._from_data(self._data.copy(deep=deep))
        if self._levels is not None:
            mi._levels = [s.copy(deep) for s in self._levels]
        if self._codes is not None:
            mi._codes = self._codes.copy(deep)
        if names is not None:
            mi.names = names
        elif self.names is not None:
            mi.names = self.names.copy()

        return mi

    def __iter__(self):
        cudf.utils.utils.raise_iteration_error(obj=self)

    def __repr__(self):
        max_seq_items = get_option("display.max_seq_items") or len(self)

        if len(self) > max_seq_items:
            n = int(max_seq_items / 2) + 1
            # TODO: Update the following two arange calls to
            # a single arange call once arange has support for
            # a vector start/end points.
            indices = column.arange(start=0, stop=n, step=1)
            indices = indices.append(
                column.arange(start=len(self) - n, stop=len(self), step=1)
            )
            preprocess = self.take(indices)
        else:
            preprocess = self.copy(deep=False)

        if any(col.has_nulls for col in preprocess._data.columns):
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
                        cudf._NA_REP
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

            if PANDAS_GE_120:
                # TODO: Remove this whole `if` block,
                # this is a workaround for the following issue:
                # https://github.com/pandas-dev/pandas/issues/39984
                preprocess_pdf = pd.DataFrame(
                    {
                        name: col.to_pandas(nullable=(col.dtype.kind != "f"))
                        for name, col in preprocess._data.items()
                    }
                )

                preprocess_pdf.columns = preprocess.names
                preprocess = pd.MultiIndex.from_frame(preprocess_pdf)
            else:
                preprocess = preprocess.to_pandas(nullable=True)
            preprocess.values[:] = tuples_list
        else:
            preprocess = preprocess.to_pandas(nullable=True)

        output = preprocess.__repr__()
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
    def codes(self):
        """
        Returns the codes of the underlying MultiIndex.

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
        >>> midx.codes
           a  b
        0  0  0
        1  1  1
        2  2  2
        """
        if self._codes is None:
            self._compute_levels_and_codes()
        return self._codes

    @property
    def nlevels(self):
        """Integer number of levels in this MultiIndex."""
        return len(self._data)

    @property
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
        [0    1
        1    2
        2    3
        dtype: int64, 0    10
        1    11
        2    12
        dtype: int64]
        """
        if self._levels is None:
            self._compute_levels_and_codes()
        return self._levels

    @property
    def ndim(self):
        """Dimension of the data. For MultiIndex ndim is always 2."""
        return 2

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
        -------
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

            res = []
            for name in self.names:
                level_idx = self.get_level_values(name)
                value_idx = values_idx.get_level_values(name)

                existence = level_idx.isin(value_idx)
                res.append(existence)

            result = res[0]
            for i in res[1:]:
                result = result & i
        else:
            level_series = self.get_level_values(level)
            result = level_series.isin(values)

        return result

    def where(self, cond, other=None, inplace=False):
        raise NotImplementedError(
            ".where is not supported for MultiIndex operations"
        )

    def _compute_levels_and_codes(self):
        levels = []

        codes = {}
        for name, col in self._data.items():
            code, cats = cudf.Series._from_data({None: col}).factorize()
            codes[name] = code.astype(np.int64)
            levels.append(cudf.Series(cats, name=None))

        self._levels = levels
        self._codes = cudf.DataFrame._from_data(codes)

    def _compute_validity_mask(self, index, row_tuple, max_length):
        """ Computes the valid set of indices of values in the lookup
        """
        lookup = cudf.DataFrame()
        for name, row in zip(index.names, row_tuple):
            if isinstance(row, slice) and row == slice(None):
                continue
            lookup[name] = cudf.Series(row)
        frame = index.to_frame(index=False)
        data_table = cudf.concat(
            [
                frame,
                cudf.DataFrame(
                    {"idx": cudf.Series(column.arange(len(frame)))}
                ),
            ],
            axis=1,
        )
        result = lookup.merge(data_table)["idx"]
        # Avoid computing levels unless the result of the merge is empty,
        # which suggests that a KeyError should be raised.
        if len(result) == 0:
            for idx, row in enumerate(row_tuple):
                if row == slice(None):
                    continue
                if row not in index.levels[idx]._column:
                    raise KeyError(row)
        return result

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
                return column.arange(start, stop, step)
            start_values = self._compute_validity_mask(
                index, row_tuple.start, max_length
            )
            stop_values = self._compute_validity_mask(
                index, row_tuple.stop, max_length
            )
            return column.arange(start_values.min(), stop_values.max() + 1)
        elif isinstance(row_tuple, numbers.Number):
            return row_tuple
        return self._compute_validity_mask(index, row_tuple, max_length)

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
                k if index.names is None else index.names[k],
                cudf.Series._from_data({None: index._data.columns[k]}),
            )

        if len(result) == 1 and size == 0 and not slice_access:
            # If the final result is one row and it was not mapped into
            # directly, return a Series with a tuple as name.
            result = result.T
            result = result[result._data.names[0]]
        elif len(result) == 0 and not slice_access:
            # Pandas returns an empty Series with a tuple as name
            # the one expected result column
            result = cudf.Series._from_data(
                {}, name=tuple((col[0] for col in index._data.columns))
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
                codes=index.codes.iloc[:, size:],
                names=index.names[size:],
            )

        if isinstance(index_key, tuple):
            result = result.set_index(index)
        return result

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

    def __eq__(self, other):
        if isinstance(other, MultiIndex):
            for self_col, other_col in zip(
                self._data.values(), other._data.values(),
            ):
                if not self_col.equals(other_col):
                    return False
            return self.names == other.names
        return NotImplemented

    @property
    def size(self):
        # The size of a MultiIndex is only dependent on the number of rows.
        return self._num_rows

    def take(self, indices):
        if isinstance(indices, (Integral, Sequence)):
            indices = np.array(indices)
        elif isinstance(indices, cudf.Series) and indices.has_nulls:
            raise ValueError("Column must have no nulls.")
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = column.arange(start, stop, step)
        result = MultiIndex.from_frame(
            self.to_frame(index=False).take(indices)
        )
        if self._codes is not None:
            result._codes = self._codes.take(indices)
        if self._levels is not None:
            result._levels = self._levels
        result.names = self.names
        return result

    def serialize(self):
        header, frames = super().serialize()
        # Overwrite the names in _data with the true names.
        header["column_names"] = pickle.dumps(self.names)
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        if "names" in header:
            warnings.warn(
                "MultiIndex objects serialized in cudf version "
                "21.10 or older will no longer be deserializable "
                "after version 21.12. Please load and resave any "
                "pickles before upgrading to version 22.02.",
                DeprecationWarning,
            )
            header["column_names"] = header["names"]
        column_names = pickle.loads(header["column_names"])
        if "source_data" in header:
            warnings.warn(
                "MultiIndex objects serialized in cudf version "
                "21.08 or older will no longer be deserializable "
                "after version 21.10. Please load and resave any "
                "pickles before upgrading to version 21.12.",
                DeprecationWarning,
            )
            df = cudf.DataFrame.deserialize(header["source_data"], frames)
            return cls.from_frame(df)._set_names(column_names)

        # Spoof the column names to construct the frame, then set manually.
        header["column_names"] = pickle.dumps(range(0, len(column_names)))
        obj = super().deserialize(header, frames)
        return obj._set_names(column_names)

    def __getitem__(self, index):
        if isinstance(index, int):
            # we are indexing into a single row of the MultiIndex,
            # return that row as a tuple:
            return self.take(index).to_pandas()[0]
        return self.take(index)

    def to_frame(self, index=True, name=None):
        # TODO: Currently this function makes a shallow copy, which is
        # incorrect. We want to make a deep copy, otherwise further
        # modifications of the resulting DataFrame will affect the MultiIndex.
        df = cudf.DataFrame._from_data(data=self._data)
        if index:
            df = df.set_index(self)
        if name is not None:
            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have the same length as "
                    "number of levels on index."
                )
            df.columns = name
        return df

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

    @classmethod
    def _concat(cls, objs):

        source_data = [o.to_frame(index=False) for o in objs]

        # TODO: Verify if this is really necesary or if we can rely on
        # DataFrame._concat.
        if len(source_data) > 1:
            colnames = source_data[0].columns
            for obj in source_data[1:]:
                obj.columns = colnames

        source_data = cudf.DataFrame._concat(source_data)
        names = [None for x in source_data.columns]
        objs = list(filter(lambda o: o.names is not None, objs))
        for o in range(len(objs)):
            for i, name in enumerate(objs[o].names):
                names[i] = names[i] or name
        return cudf.MultiIndex.from_frame(source_data, names=names)

    @classmethod
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

    @property
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

    @property
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
        <class 'cupy.core.core.ndarray'>
        """
        return self.to_frame(index=False).values

    @classmethod
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

        ilevels = sorted([self._level_index_from_level(lev) for lev in level])

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
        Int64Index([0, 1, 2, 0, 1, 2], dtype='int64', name='third')
        """
        mi = self.copy(deep=False)
        mi._poplevels(level)
        if mi.nlevels == 1:
            return mi.get_level_values(mi.names[0])
        else:
            return mi

    def to_pandas(self, nullable=False, **kwargs):
        result = self.to_frame(index=False).to_pandas(nullable=nullable)
        return pd.MultiIndex.from_frame(result, names=self.names)

    @classmethod
    def from_pandas(cls, multiindex, nan_as_null=None):
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

        # if `multiindex` has two or more levels that
        # have the same name, then `multiindex.to_frame()`
        # results in a DataFrame containing only one of those
        # levels. Thus, set `names` to some tuple of unique values
        # and then call `multiindex.to_frame(name=names)`,
        # which preserves all levels of `multiindex`.
        names = tuple(range(len(multiindex.names)))

        df = cudf.DataFrame.from_pandas(
            multiindex.to_frame(index=False, name=names), nan_as_null
        )
        return cls.from_frame(df, names=multiindex.names)

    @cached_property
    def is_unique(self):
        return len(self) == len(self.unique())

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._is_sorted(ascending=None, null_position=None)

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        return self._is_sorted(
            ascending=[False] * len(self.levels), null_position=None
        )

    def argsort(self, ascending=True, **kwargs):
        return self._get_sorted_inds(ascending=ascending, **kwargs).values

    def sort_values(self, return_indexer=False, ascending=True, key=None):
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")

        indices = cudf.Series._from_data(
            {None: self._get_sorted_inds(ascending=ascending)}
        )
        index_sorted = as_index(self.take(indices), name=self.names)

        if return_indexer:
            return index_sorted, cupy.asarray(indices)
        else:
            return index_sorted

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

    def unique(self):
        return self.drop_duplicates(ignore_index=True)

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in MultiIndex object
        to `<NA>` as a preprocessing step to `__repr__` methods.
        """
        index_df = self.to_frame(index=False)
        return MultiIndex.from_frame(
            index_df._clean_nulls_from_dataframe(index_df), names=self.names
        )

    def memory_usage(self, deep=False):
        n = 0
        for col in self._data._columns:
            n += col._memory_usage(deep=deep)
        if self._levels:
            for level in self._levels:
                n += level.memory_usage(deep=deep)
        if self._codes:
            for col in self._codes._columns:
                n += col._memory_usage(deep=deep)
        return n

    def difference(self, other, sort=None):
        if hasattr(other, "to_pandas"):
            other = other.to_pandas()
        return self.to_pandas().difference(other, sort)

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

    def get_loc(self, key, method=None, tolerance=None):
        """
        Get location for a label or a tuple of labels.

        The location is returned as an integer/slice or boolean mask.

        Parameters
        ----------
        key : label or tuple of labels (one for each level)
        method : None

        Returns
        -------
        loc : int, slice object or boolean mask
            - If index is unique, search result is unique, return a single int.
            - If index is monotonic, index is returned as a slice object.
            - Otherwise, cudf attempts a best effort to convert the search
              result into a slice object, and will return a boolean mask if
              failed to do so. Notice this can deviate from Pandas behavior
              in some situations.

        Examples
        --------
        >>> import cudf
        >>> mi = cudf.MultiIndex.from_tuples(
            [('a', 'd'), ('b', 'e'), ('b', 'f')])
        >>> mi.get_loc('b')
        slice(1, 3, None)
        >>> mi.get_loc(('b', 'e'))
        1
        >>> non_monotonic_non_unique_idx = cudf.MultiIndex.from_tuples(
            [('c', 'd'), ('b', 'e'), ('a', 'f'), ('b', 'e')])
        >>> non_monotonic_non_unique_idx.get_loc('b') # differ from pandas
        slice(1, 4, 2)

        .. pandas-compat::
            **MultiIndex.get_loc**

            The return types of this function may deviates from the
            method provided by Pandas. If the index is neither
            lexicographically sorted nor unique, a best effort attempt is made
            to coerce the found indices into a slice. For example:

            .. code-block::

                >>> import pandas as pd
                >>> import cudf
                >>> x = pd.MultiIndex.from_tuples(
                            [(2, 1, 1), (1, 2, 3), (1, 2, 1),
                                (1, 1, 1), (1, 1, 1), (2, 2, 1)]
                        )
                >>> x.get_loc(1)
                array([False,  True,  True,  True,  True, False])
                >>> cudf.from_pandas(x).get_loc(1)
                slice(1, 5, 1)
        """
        if tolerance is not None:
            raise NotImplementedError(
                "Parameter tolerance is unsupported yet."
            )
        if method is not None:
            raise NotImplementedError(
                "only the default get_loc method is currently supported for"
                " MultiIndex"
            )

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )
        is_unique = self.is_unique
        key = (key,) if not isinstance(key, tuple) else key

        # Handle partial key search. If length of `key` is less than `nlevels`,
        # Only search levels up to `len(key)` level.
        key_as_table = libcudf.table.Table(
            {i: column.as_column(k, length=1) for i, k in enumerate(key)}
        )
        partial_index = self.__class__._from_data(
            data=self._data.select_by_index(slice(key_as_table._num_columns))
        )
        (lower_bound, upper_bound, sort_inds,) = _lexsorted_equal_range(
            partial_index, key_as_table, is_sorted
        )

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

        true_inds = cupy.array(
            sort_inds.slice(lower_bound, upper_bound).to_gpu_array()
        )
        true_inds = _maybe_indices_to_slice(true_inds)
        if isinstance(true_inds, slice):
            return true_inds

        # Not sorted and not unique. Return a boolean mask
        mask = cupy.full(self._data.nrows, False)
        mask[true_inds] = True
        return mask
