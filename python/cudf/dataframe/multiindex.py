# Copyright (c) 2019, NVIDIA CORPORATION.

import pandas as pd
import numpy as np

from collections.abc import Sequence
from copy import copy, deepcopy

from cudf.dataframe import columnops
from cudf.comm.serialize import register_distributed_serializer
import cudf.dataframe.index as indexPackage
from cudf.dataframe.index import StringIndex
from numba.cuda.cudadrv.devicearray import DeviceNDArray


class MultiIndex(indexPackage.Index):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.

    Properties
    ---
    levels: Labels for each category in the index hierarchy.
    codes: Assignment of individual items into the categories of the hierarchy.
    names: Name for each level
    """

    def __init__(self, levels, codes, names=None):
        if isinstance(names, Sequence):
            if all(x is None for x in names):
                names = None
        if names is None:
            column_names = list(range(len(codes)))
        else:
            column_names = names
        if len(codes) == 0:
            raise ValueError('MultiIndex codes can not be empty.')
        from cudf import DataFrame
        if not isinstance(codes, DataFrame) and\
                not isinstance(codes[0], (Sequence,
                               pd.core.indexes.frozen.FrozenNDArray)):
            raise TypeError('Codes is not a Sequence of sequences')
        if not isinstance(codes, DataFrame):
            self.codes = DataFrame()
            for idx, code in enumerate(codes):
                self.codes.add_column(column_names[idx],
                                      columnops.as_column(code))
        else:
            self.codes = codes
        self.levels = levels
        self._validate_levels_and_codes(self.levels, self.codes)
        self.name = None
        self.names = names

    def _validate_levels_and_codes(self, levels, codes):
        levels = np.array(levels)
        if len(levels) != len(codes.columns):
            raise ValueError('MultiIndex has unequal number of levels and '
                             'codes and is inconsistent!')
        code_length = len(codes[codes.columns[0]])
        for index, code in enumerate(codes):
            if code_length != len(codes[code]):
                raise ValueError('MultiIndex length of codes does not match '
                                 'and is inconsistent!')
        for index, code in enumerate(codes):
            if codes[code].max() > len(levels[index])-1:
                raise ValueError('MultiIndex code %d contains value %d larger '
                                 'than maximum level size at this position')

    def copy(self, deep=True):
        if(deep):
            result = deepcopy(self)
        else:
            result = copy(self)
        result.name = self.name
        return result

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
        result.names = self.names[n:]
        return result

    def __repr__(self):
        return "MultiIndex(levels=" + str(self.levels) +\
               ",\ncodes=" + str(self.codes) + ")"

    def __getitem__(self, index):
        if isinstance(index, slice):
            # a new MultiIndex with the sliced codes, same levels and names
            None
        elif isinstance(index, int):
            # a tuple of the labels of the item defined by the set of codes
            # at the ith positionjA
            None
        elif isinstance(index, (list, np.ndarray)):
            result = MultiIndex(self.levels, self.codes.iloc[index])
            result.names = self.names
            return result
        if isinstance(index, (DeviceNDArray)):
            return self.take(index)
        else:
            raise IndexError('only integers, slices (`:`), ellipsis (`...`),'
            'numpy.newaxis (`None`) and integer or boolean arrays are valid '
            'indices')  # noqa: E128

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
            if index_of_code_at_level == None:
                raise KeyError(element)
            matches = []
            for k, code in enumerate(self.codes[self.codes.columns[i]]):
                if k in validity_mask or len(validity_mask) == 0:
                    if code == index_of_code_at_level:
                        matches.append(k)
            # matches = self.codes.map([j if c[i] == index_of_code_at_level\
            #       else None for j,c in codes])
            if len(matches) != 0:
                validity_mask = matches
        return validity_mask

    def get_row_major(self, df, row_tuple):
        valid_indices = self._compute_validity_mask(df, row_tuple)
        from cudf import Series
        result = df.take(Series(valid_indices))
        # Build new index - INDEX based MultiIndex
        # ---------------
        from cudf import DataFrame
        out_index = DataFrame()
        # Select the last n-k columns where n is the number of source
        # levels and k is the length of the indexing tuple
        for k in range(len(row_tuple), len(df.index.levels)):
            out_index.add_column(df.index.names[k],
                                 df.index.codes[df.index.codes.columns[k]])
        # If there's only one column remaining in the output index, convert
        # it into a StringIndex and name the final index values according
        # to the proper codes.
        if len(out_index.columns) == 1:
            out_index = []
            for val in result.index.codes[result.index.codes.columns[len(result.index.codes.columns)-1]]:  # noqa: E501
                out_index.append(result.index.levels[
                        len(result.index.codes.columns)-1][val])
            # TODO: Warning! The final index column could be arbitrarily
            # ordered integers, not Strings, so we need to check for that
            # dtype and produce a GenericIndex instead of a StringIndex
            out_index = StringIndex(out_index)
            out_index.name = result.index.names[len(result.index.names)-1]
            result.index = out_index
        else:
            # Otherwise pop the leftmost levels, names, and codes from the
            # source index until it has the correct number of columns (n-k)
            if(len(out_index.columns)) > 0:
                result.reset_index(drop=True)
                result.index = result.index._popn(len(row_tuple))
        # Finally, if n-k==1 and only 1 row is returned, convert the
        # resulting DataFrame into a Series instead.
        """
        print('result')
        print(len(result))
        print('n-k')
        print(len(row_tuple)-len(self.levels))
        if len(result) == 1 and len(self.levels)-len(row_tuple) == 1:
            from cudf.dataframe import Series
            result = Series(result.iloc[0])
            result.name = row_tuple
        """
        return result

    def get_column_major(self, df, row_tuple):
        valid_indices = self._compute_validity_mask(df, row_tuple)
        from cudf import DataFrame
        result = DataFrame()
        for ix, col in enumerate(df.columns):
            if ix in valid_indices:
                result[ix] = df._cols[ix]
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
        return len(self.codes[self.codes.columns[0]])

    def __eq__(self, other):
        return self.levels == other.levels and\
                self.codes == other.codes and\
                self.names == other.names

    @property
    def is_contiguous(self):
        return True

    @property
    def size(self):
        return len(self.codes[0])

    def take(self, indices):
        from collections.abc import Sequence
        from cudf import Series
        from numbers import Integral
        if isinstance(indices, (Integral, Sequence)):
            indices = np.array(indices)
        elif isinstance(indices, Series):
            indices = indices.to_gpu_array()
        codes = self.codes.take(indices)
        result = MultiIndex(self.levels, codes)
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
        result = []
        for level, item in enumerate(match.codes):
            result.append(match.levels[level][match.codes[item][0]])
        return tuple(result)

    @property
    def _values(self):
        return list([i for i in self])

    @classmethod
    def from_tuples(cls, tuples, names=None):
        # cheating
        pdi = pd.MultiIndex.from_tuples(tuples, names=names)
        print(pdi)
        result = cls.from_pandas(pdi)
        return result

    @classmethod
    def from_frame(cls, dataframe, names=None):
        # cheating
        pdi = pd.MultiIndex.from_frame(dataframe.to_pandas(), names=names)
        result = cls.from_pandas(pdi)
        return result

    @classmethod
    def from_product(cls, arrays, names=None):
        # cheating
        pdi = pd.MultiIndex.from_product(arrays, names=names)
        result = cls.from_pandas(pdi)
        return result

    def to_pandas(self):
        pandas_codes = []
        for code in self.codes.columns:
            pandas_codes.append(self.codes[code].to_array())
        return pd.MultiIndex(levels=self.levels, codes=pandas_codes,
                             names=self.names)

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
            raise TypeError('not a pandas.MultiIndex')

        mi = cls(levels=multiindex.levels,
                 codes=multiindex.codes,
                 names=multiindex.names)
        return mi


register_distributed_serializer(MultiIndex)
