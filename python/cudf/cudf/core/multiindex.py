# Copyright (c) 2019, NVIDIA CORPORATION.

import numbers
import pickle
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

import cudf._lib as libcudf
from cudf.core.column import column
from cudf.core.index import Index, as_index
from cudf.utils import cudautils


class MultiIndex(Index):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Properties
    ---
    levels: Labels for each category in the index hierarchy.
    codes: Assignment of individual items into the categories of the hierarchy.
    names: Name for each level
    """

    def __init__(
        self, levels=None, codes=None, labels=None, names=None, **kwargs
    ):
        from cudf.core.series import Series

        self.name = None
        self.names = names
        self._source_data = None
        column_names = []
        if labels:
            warnings.warn(
                "the 'labels' keyword is deprecated, use 'codes' " "instead",
                FutureWarning,
            )
        if labels and not codes:
            codes = labels

        # early termination enables lazy evaluation of codes
        if "source_data" in kwargs:
            self._source_data = kwargs["source_data"].reset_index(drop=True)
            self._codes = codes
            self._levels = levels
            return

        # name setup
        if isinstance(
            names,
            (
                Sequence,
                pd.core.indexes.frozen.FrozenNDArray,
                pd.core.indexes.frozen.FrozenList,
            ),
        ):
            if sum(x is None for x in names) > 1:
                column_names = list(range(len(codes)))
            else:
                column_names = names
        elif names is None:
            column_names = list(range(len(codes)))
        else:
            column_names = names

        if len(levels) == 0:
            raise ValueError("Must pass non-zero number of levels/codes")

        from cudf import DataFrame

        if not isinstance(codes, DataFrame) and not isinstance(
            codes[0], (Sequence, pd.core.indexes.frozen.FrozenNDArray)
        ):
            raise TypeError("Codes is not a Sequence of sequences")

        if isinstance(codes, DataFrame):
            self._codes = codes
        elif len(levels) == len(codes):
            self._codes = DataFrame()
            for i, codes in enumerate(codes):
                name = column_names[i] or i
                codes = column.as_column(codes)
                self._codes[name] = codes.astype(np.int64)
        else:
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )

        self._levels = [Series(level) for level in levels]
        self._validate_levels_and_codes(self._levels, self._codes)

        self._source_data = DataFrame()
        for i, name in enumerate(self._codes.columns):
            codes = as_index(self._codes[name]._column)
            if -1 in self._codes[name].values:
                # Must account for null(s) in _source_data column
                level = DataFrame(
                    {name: [None] + list(self._levels[i])},
                    index=range(-1, len(self._levels[i])),
                )
            else:
                level = DataFrame({name: self._levels[i]})
            level = DataFrame(index=codes).join(level)
            self._source_data[name] = level[name].reset_index(drop=True)

        self.names = [None] * len(self._levels) if names is None else names

    def _validate_levels_and_codes(self, levels, codes):
        if len(levels) != len(codes.columns):
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )
        code_length = len(codes[codes.columns[0]])
        for index, code in enumerate(codes):
            if code_length != len(codes[code]):
                raise ValueError(
                    "MultiIndex length of codes does not match "
                    "and is inconsistent!"
                )
        for index, code in enumerate(codes):
            if codes[code].max() > len(levels[index]) - 1:
                raise ValueError(
                    "MultiIndex code %d contains value %d larger "
                    "than maximum level size at this position"
                )

    def copy(self, deep=True):
        mi = MultiIndex(source_data=self._source_data.copy(deep))
        if self._levels is not None:
            mi._levels = [s.copy(deep) for s in self._levels]
        if self._codes is not None:
            mi._codes = self._codes.copy(deep)
        if self.names is not None:
            mi.names = self.names.copy()
        return mi

    def deepcopy(self):
        return self.copy(deep=True)

    def __copy__(self):
        return self.copy(deep=True)

    def _popn(self, n):
        """ Returns a copy of this index without the left-most n values.

        Removes n names, labels, and codes in order to build a new index
        for results.
        """
        from cudf import DataFrame

        codes = DataFrame()
        for idx in self.codes.columns[n:]:
            codes.add_column(idx, self.codes[idx])
        result = MultiIndex(self.levels[n:], codes)
        if self.names is not None:
            result.names = self.names[n:]
        return result

    def __repr__(self):
        return (
            "MultiIndex(levels="
            + str(self.levels)
            + ",\ncodes="
            + str(self.codes)
            + ")"
        )

    @property
    def codes(self):
        if self._codes is None:
            self._compute_levels_and_codes()
        return self._codes

    @property
    def levels(self):
        if self._levels is None:
            self._compute_levels_and_codes()
        return self._levels

    @property
    def labels(self):
        warnings.warn(
            "This feature is deprecated in pandas and will be"
            "dropped from cudf as well.",
            FutureWarning,
        )
        return self.codes

    def _compute_levels_and_codes(self):
        levels = []
        from cudf import DataFrame

        codes = DataFrame()
        for name in self._source_data.columns:
            code, cats = self._source_data[name].factorize()
            codes[name] = code.reset_index(drop=True).astype(np.int64)
            cats.name = None
            cats = cats.reset_index(drop=True)._copy_construct(name=None)
            levels.append(cats)

        self._levels = levels
        self._codes = codes

    def _compute_validity_mask(self, index, row_tuple, max_length):
        """ Computes the valid set of indices of values in the lookup
        """
        from cudf import DataFrame
        from cudf import Series
        from cudf import concat
        from cudf.utils.cudautils import arange

        lookup = DataFrame()
        for idx, row in enumerate(row_tuple):
            if row == slice(None):
                continue
            lookup[index._source_data.columns[idx]] = Series(row)
        data_table = concat(
            [
                index._source_data,
                DataFrame({"idx": Series(arange(len(index._source_data)))}),
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
        from cudf.utils.cudautils import arange
        from cudf import Series

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
                return arange(start, stop, step)
            start_values = self._compute_validity_mask(
                index, row_tuple.start, max_length
            )
            stop_values = self._compute_validity_mask(
                index, row_tuple.stop, max_length
            )
            return Series(arange(start_values.min(), stop_values.max() + 1))
        elif isinstance(row_tuple, numbers.Number):
            return row_tuple
        return self._compute_validity_mask(index, row_tuple, max_length)

    def _index_and_downcast(self, result, index, index_key):
        from cudf import DataFrame
        from cudf import Series

        if isinstance(index_key, (numbers.Number, slice)):
            index_key = [index_key]
        if (
            len(index_key) > 0 and not isinstance(index_key, tuple)
        ) or isinstance(index_key[0], slice):
            index_key = index_key[0]

        slice_access = False
        if isinstance(index_key, slice):
            slice_access = True
        out_index = DataFrame()
        # Select the last n-k columns where n is the number of _source_data
        # columns and k is the length of the indexing tuple
        size = 0
        if not isinstance(index_key, (numbers.Number, slice)):
            size = len(index_key)
        for k in range(size, len(index._source_data.columns)):
            out_index.add_column(
                index.names[k],
                index._source_data[index._source_data.columns[k]],
            )

        if len(result) == 1 and size == 0 and slice_access is False:
            # If the final result is one row and it was not mapped into
            # directly, return a Series with a tuple as name.
            result = result.T
            result = result[result.columns[0]]
        elif len(result) == 0 and slice_access is False:
            # Pandas returns an empty Series with a tuple as name
            # the one expected result column
            series_name = []
            for idx, code in enumerate(index._source_data.columns):
                series_name.append(index._source_data[code][0])
            result = Series([])
            result.name = tuple(series_name)
        elif len(out_index.columns) == 1:
            # If there's only one column remaining in the output index, convert
            # it into an Index and name the final index values according
            # to the _source_data column names
            last_column = index._source_data.columns[-1]
            out_index = index._source_data[last_column]
            out_index = as_index(out_index)
            out_index.name = index.names[len(index.names) - 1]
            index = out_index
        elif len(out_index.columns) > 1:
            # Otherwise pop the leftmost levels, names, and codes from the
            # source index until it has the correct number of columns (n-k)
            result.reset_index(drop=True)
            index = index._popn(size)
        if isinstance(index_key, tuple):
            result = result.set_index(index)
        return result

    def _get_row_major(self, df, row_tuple):
        from cudf import Series

        valid_indices = self._get_valid_indices_by_tuple(
            df.index, row_tuple, len(df.index)
        )
        indices = Series(valid_indices)
        result = df.take(indices)
        final = self._index_and_downcast(result, result.index, row_tuple)
        return final

    def _get_column_major(self, df, row_tuple):
        from cudf import Series
        from cudf import DataFrame

        valid_indices = self._get_valid_indices_by_tuple(
            df.columns, row_tuple, len(df._cols)
        )
        result = df._take_columns(valid_indices)
        if isinstance(row_tuple, (numbers.Number, slice)):
            row_tuple = [row_tuple]
        if len(result) == 0 and len(result.columns) == 0:
            result_columns = df.columns.copy(deep=False)
            clear_codes = DataFrame()
            for name in df.columns.names:
                clear_codes[name] = Series([])
            result_columns._codes = clear_codes
            result_columns._source_data = clear_codes
            result.columns = result_columns
        elif len(row_tuple) < len(self.levels) and (
            not slice(None) in row_tuple
            and not isinstance(row_tuple[0], slice)
        ):
            columns = self._popn(len(row_tuple))
            result.columns = columns.take(valid_indices)
        else:
            result.columns = self.take(valid_indices)
        if len(result.columns.levels) == 1:
            columns = []
            for code in result.columns.codes[result.columns.codes.columns[0]]:
                columns.append(result.columns.levels[0][code])
            name = result.columns.names[0]
            result.columns = as_index(columns, name=name)
        if len(row_tuple) == len(self.levels) and len(result.columns) == 1:
            result = list(result._cols.values())[0]
        return result

    def _split_tuples(self, tuples):
        if len(tuples) == 1:
            return tuples, slice(None)
        elif isinstance(tuples[0], tuple):
            row = tuples[0]
            if len(tuples) == 1:
                column = slice(None)
            else:
                column = tuples[1]
            return row, column
        elif isinstance(tuples[0], slice):
            return tuples
        else:
            return tuples, slice(None)

    def __len__(self):
        return len(self._source_data)

    def equals(self, other):
        if self is other:
            return True
        if len(self) != len(other):
            return False
        return self == other

    def __eq__(self, other):
        if not hasattr(other, "_levels"):
            return False
        # Lazy comparison
        if isinstance(other, MultiIndex) or hasattr(other, "_source_data"):
            return self._source_data.equals(other._source_data)
        else:
            # Lazy comparison isn't possible - MI was created manually.
            # Actually compare the MI, not its source data (it doesn't have
            # any).
            equal_levels = self.levels == other.levels
            if isinstance(equal_levels, np.ndarray):
                equal_levels = equal_levels.all()
            return (
                equal_levels
                and self.codes.equals(other.codes)
                and self.names == other.names
            )

    @property
    def is_contiguous(self):
        return True

    @property
    def size(self):
        return len(self._source_data)

    def take(self, indices):
        from collections.abc import Sequence
        from cudf import Series
        from numbers import Integral

        if isinstance(indices, (Integral, Sequence)):
            indices = np.array(indices)
        elif isinstance(indices, Series):
            if indices.null_count != 0:
                raise ValueError("Column must have no nulls.")
            indices = indices.data.mem
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = cudautils.arange(start, stop, step)
        result = MultiIndex(source_data=self._source_data.take(indices))
        if self._codes is not None:
            result._codes = self._codes.take(indices)
        if self._levels is not None:
            result._levels = self._levels
        result.names = self.names
        return result

    def serialize(self):
        """Serialize into pickle format suitable for file storage or network
        transmission.
        """
        header = {}
        header["type"] = pickle.dumps(type(self))
        header["names"] = pickle.dumps(self.names)

        header["source_data"], frames = self._source_data.serialize()

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        """Convert from pickle format into Index
        """
        names = pickle.loads(header["names"])

        source_data_typ = pickle.loads(header["source_data"]["type"])
        source_data = source_data_typ.deserialize(
            header["source_data"], frames
        )

        names = pickle.loads(header["names"])
        return MultiIndex(names=names, source_data=source_data)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.codes):
            result = self[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, index):
        # TODO: This should be a take of the _source_data only
        match = self.take(index)
        if isinstance(index, slice):
            return match
        result = []
        for level, item in enumerate(match.codes):
            result.append(match.levels[level][match.codes[item][0]])
        return tuple(result)

    def to_frame(self, index=True, name=None):
        df = self._source_data
        if index:
            df = df.set_index(self)
        if name:
            if len(name) != len(self.levels):
                raise ValueError(
                    "'name' should have th same length as "
                    "number of levels on index."
                )
            df.columns = name
        return df

    def get_level_values(self, level):
        colnames = list(self._source_data.columns)
        if level not in colnames:
            if isinstance(level, int):
                if level < 0:
                    level = level + len(colnames)
                if level < 0 or level >= len(colnames):
                    raise IndexError(f"Invalid level number: '{level}'")
                level_idx = level
                level = colnames[level_idx]
            else:
                raise KeyError(f"Level not found: '{level}'")
        level_values = self._source_data[level]
        return level_values

    def _to_frame(self):
        from cudf import DataFrame, Series

        # for each column of codes
        # replace column with mapping from integers to levels
        df = self.codes.copy(deep=False)
        for idx, col in enumerate(df.columns):
            # use merge as a replace fn
            level = DataFrame(
                {
                    "idx": Series(
                        cudautils.arange(
                            len(self.levels[idx]), dtype=df[col].dtype
                        )
                    ),
                    "level": self.levels[idx],
                }
            )
            code = DataFrame({"idx": df[col]})
            df[col] = code.merge(level).level
        return df

    @property
    def _values(self):
        return list([i for i in self])

    @classmethod
    def _concat(cls, objs):
        from cudf import DataFrame, MultiIndex

        source_data = [o._source_data for o in objs]
        source_data = DataFrame._concat(source_data)
        names = [None for x in source_data.columns]
        objs = list(filter(lambda o: o.names is not None, objs))
        for o in range(len(objs)):
            for i, name in enumerate(objs[o].names):
                names[i] = names[i] or name
        return MultiIndex(names=names, source_data=source_data)

    @classmethod
    def from_tuples(cls, tuples, names=None):
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_tuples(tuples, names=names)
        result = cls.from_pandas(pdi)
        return result

    @classmethod
    def from_frame(cls, dataframe, names=None):
        return cls(source_data=dataframe, names=names)

    @classmethod
    def from_product(cls, arrays, names=None):
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_product(arrays, names=names)
        result = cls.from_pandas(pdi)
        return result

    def to_pandas(self):
        pandas_codes = []
        for code in self.codes.columns:
            pandas_codes.append(self.codes[code].to_array())

        # We do two things here to mimic Pandas behavior:
        # 1. as_index() on each level, so DatetimeColumn becomes DatetimeIndex
        # 2. convert levels to numpy array so empty levels become Float64Index
        levels = np.array(
            [as_index(level).to_pandas() for level in self.levels]
        )

        # Backwards compatibility:
        # Construct a dummy MultiIndex and check for the codes attr.
        # This indicates that it is pandas >= 0.24
        # If no codes attr is present it is pandas <= 0.23
        if hasattr(pd.MultiIndex([[]], [[]]), "codes"):
            pandas_mi = pd.MultiIndex(levels=levels, codes=pandas_codes)
        else:
            pandas_mi = pd.MultiIndex(levels=levels, labels=pandas_codes)
        if self.names is not None:
            pandas_mi.names = self.names
        return pandas_mi

    @classmethod
    def from_pandas(cls, multiindex):
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
                                codes=[[0, 1], [1, ]])
        >>> cudf.from_pandas(pmi)
        MultiIndex( ... )
        """
        if not isinstance(multiindex, pd.MultiIndex):
            raise TypeError("not a pandas.MultiIndex")

        if hasattr(multiindex, "codes"):
            mi = cls(
                levels=multiindex.levels,
                codes=multiindex.codes,
                names=multiindex.names,
            )
        else:
            mi = cls(
                levels=multiindex.levels,
                codes=multiindex.labels,
                names=multiindex.names,
            )
        return mi

    @property
    def is_unique(self):
        if not hasattr(self, "_is_unique"):
            self._is_unique = (
                self._source_data._size
                == self._source_data.drop_duplicates()._size
            )
        return self._is_unique

    @property
    def is_monotonic_increasing(self):
        if not hasattr(self, "_is_monotonic_increasing"):
            self._is_monotonic_increasing = libcudf.issorted.issorted(
                self._source_data._columns
            )
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = libcudf.issorted.issorted(
                self._source_data._columns, [1] * len(self.levels)
            )
        return self._is_monotonic_decreasing

    def repeat(self, repeats, axis=None):
        assert axis in (None, 0)
        return MultiIndex.from_frame(
            self._source_data.repeat(repeats), names=self.names
        )

    def memory_usage(self, deep=False):
        n = 0
        for col in self._source_data._columns:
            n += col._memory_usage(deep=deep)
        if self._levels:
            for level in self._levels:
                n += level.memory_usage(deep=deep)
        if self._codes:
            for col in self._codes._columns:
                n += col._memory_usage(deep=deep)
        return n
