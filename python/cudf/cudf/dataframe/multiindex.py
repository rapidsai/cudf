# Copyright (c) 2019, NVIDIA CORPORATION.

import numbers
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from cudf.comm.serialize import register_distributed_serializer
from cudf.dataframe import columnops
from cudf.dataframe.index import Index, StringIndex, as_index
from cudf.utils import cudautils, utils


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
        from cudf.dataframe.series import Series

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
            if names is None:
                self.names = self._source_data.columns
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
                codes = columnops.as_column(codes)
                self._codes[name] = codes.astype(np.int64)
        else:
            raise ValueError(
                "MultiIndex has unequal number of levels and "
                "codes and is inconsistent!"
            )

        # converting levels to numpy array will produce a Float64Index
        # (on empty levels)for levels mimicking the behavior of Pandas
        self._levels = np.array([Series(level) for level in levels])
        self._validate_levels_and_codes(self._levels, self._codes)

        self._source_data = DataFrame()
        for i, name in enumerate(self._codes.columns):
            codes = as_index(self._codes[name]._column)
            level = DataFrame({name: self._levels[i]})
            level = DataFrame(index=codes).join(level)
            self._source_data[name] = level[name].reset_index(drop=True)

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
            mi._levels = np.array([s.copy(deep) for s in self._levels])
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
        # Note: This is an O(N^2) solution using gpu masking
        # to compute new codes for the MultiIndex. There may be
        # a faster solution that could be executed on gpu at the same
        # time the groupby is calculated.
        for name in self._source_data.columns:
            code, cats = self._source_data[name].factorize()
            codes[name] = code.reset_index(drop=True).astype(np.int64)
            levels.append(cats.reset_index(drop=True))

        self._levels = levels
        self._codes = codes

    def _compute_validity_mask(self, df, row_tuple):
        """ Computes the valid set of indices of values in the lookup
        """
        validity_mask = []
        for i, element in enumerate(row_tuple):
            index_of_code_at_level = None
            for level_index in range(len(self.levels[i])):
                if self.levels[i][level_index] == element:
                    index_of_code_at_level = level_index
                    break
            if index_of_code_at_level is None:
                raise KeyError(element)
            matches = []
            for k, code in enumerate(self.codes[self.codes.columns[i]]):
                if k in validity_mask or len(validity_mask) == 0:
                    if code == index_of_code_at_level:
                        matches.append(k)
            if len(matches) != 0:
                validity_mask = matches
        return validity_mask

    def _get_row_major(self, df, row_tuple):
        slice_access = False
        if isinstance(row_tuple[0], numbers.Number):
            valid_indices = row_tuple[0]
        elif isinstance(row_tuple[0], slice):
            # 1. empty slice compute
            if row_tuple[0].stop == 0:
                valid_indices = []
            else:
                slice_access = True
                start = row_tuple[0].start or 0
                stop = row_tuple[0].stop or len(df)
                step = row_tuple[0].step or 1
                valid_indices = cudautils.arange(start, stop, step)
        else:
            valid_indices = self._compute_validity_mask(df, row_tuple)
        from cudf import Series

        result = df.take(Series(valid_indices))
        # Build new index - INDEX based MultiIndex
        # ---------------
        from cudf import DataFrame

        out_index = DataFrame()
        # Select the last n-k columns where n is the number of source
        # levels and k is the length of the indexing tuple
        size = 0
        if not isinstance(row_tuple[0], (numbers.Number, slice)):
            size = len(row_tuple)
        for k in range(size, len(df.index.levels)):
            out_index.add_column(
                df.index.names[k], df.index.codes[df.index.codes.columns[k]]
            )
        # If there's only one column remaining in the output index, convert
        # it into an Index and name the final index values according
        # to the proper codes.
        if len(out_index.columns) == 1:
            out_index = []
            for val in result.index.codes[
                result.index.codes.columns[len(result.index.codes.columns) - 1]
            ]:
                out_index.append(
                    result.index.levels[len(result.index.codes.columns) - 1][
                        val
                    ]
                )
            out_index = as_index(out_index)
            out_index.name = result.index.names[len(result.index.names) - 1]
            result.index = out_index
        else:
            if len(result) == 1 and size == 0 and slice_access is False:
                # If the final result is one row and it was not mapped into
                # directly
                result = result.T
                result = result[result.columns[0]]
                # convert to Series
                series_name = []
                for idx, code in enumerate(result.columns.codes):
                    series_name.append(
                        result.columns.levels[idx][
                            result.columns.codes[code][0]
                        ]
                    )
                result = Series(
                    list(result._cols.values())[0], name=series_name
                )
                result.name = tuple(series_name)
            elif (len(out_index.columns)) > 0:
                # Otherwise pop the leftmost levels, names, and codes from the
                # source index until it has the correct number of columns (n-k)
                result.reset_index(drop=True)
                result.index = result.index._popn(size)
        return result

    def _get_column_major(self, df, row_tuple):
        valid_indices = self._compute_validity_mask(df, row_tuple)
        from cudf import DataFrame

        result = DataFrame()
        for ix, col in enumerate(df.columns):
            if ix in valid_indices:
                result[ix] = list(df._cols.values())[ix]
        # Build new index - COLUMN based MultiIndex
        # ---------------
        if len(row_tuple) < len(self.levels):
            columns = self._popn(len(row_tuple))
            result.columns = columns.take(valid_indices)
        else:
            result.columns = self.take(valid_indices)
        if len(result.columns.levels) == 1:
            columns = []
            for code in result.columns.codes[result.columns.codes.columns[0]]:
                columns.append(result.columns.levels[0][code])
            name = result.columns.names[0]
            result.columns = StringIndex(columns, name=name)
        return result

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
            indices = indices.to_gpu_array()
        elif isinstance(indices, slice):
            start, stop, step, sln = utils.standard_python_slice(
                len(self), indices
            )
            indices = cudautils.arange(start, stop, step)
        result = MultiIndex(source_data=self._source_data.take(indices))
        result.names = self.names
        return result

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
        for idx, column in enumerate(df.columns):
            # use merge as a replace fn
            level = DataFrame(
                {
                    "idx": Series(
                        cudautils.arange(
                            len(self.levels[idx]), dtype=df[column].dtype
                        )
                    ),
                    "level": self.levels[idx],
                }
            )
            code = DataFrame({"idx": df[column]})
            df[column] = code.merge(level).level
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
        # Use Pandas for handling Python host objects
        pdi = pd.MultiIndex.from_frame(dataframe.to_pandas(), names=names)
        result = cls.from_pandas(pdi)
        return result

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
        # Backwards compatibility:
        # Construct a dummy MultiIndex and check for the codes attr.
        # This indicates that it is pandas >= 0.24
        # If no codes attr is present it is pandas <= 0.23
        if hasattr(pd.MultiIndex([[]], [[]]), "codes"):
            return pd.MultiIndex(
                levels=self.levels, codes=pandas_codes, names=self.names
            )
        else:
            return pd.MultiIndex(
                levels=self.levels, labels=pandas_codes, names=self.names
            )

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
            self._is_monotonic_increasing = self._source_data.argsort(
                ascending=True
            ).is_monotonic_increasing
        return self._is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        if not hasattr(self, "_is_monotonic_decreasing"):
            self._is_monotonic_decreasing = self._source_data.argsort(
                ascending=True
            ).is_monotonic_decreasing
        return self._is_monotonic_decreasing


register_distributed_serializer(MultiIndex)
