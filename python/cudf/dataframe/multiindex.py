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
        self.levels = levels
        from cudf import DataFrame
        if names is None:
            column_names = list(range(len(codes)))
        else:
            column_names = names
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
        self.name = None
        self.names = names
        self._validate_levels_and_codes(self.levels, self.codes)

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

    def get(self, df, row_tuple):
        # Assume row is a tuple
        validity_mask = []
        for i, element in enumerate(row_tuple):
            # index_of_code_at_level = np.where(self.levels[i] == row_tuple[i])[0][0]  # noqa: E501
            for level_index in range(len(self.levels)):
                if self.levels[i][level_index] == row_tuple[i]:
                    index_of_code_at_level = level_index
            matches = []
            for k, code in enumerate(self.codes[self.codes.columns[i]]):
                if k in validity_mask or len(validity_mask) == 0:
                    if code == index_of_code_at_level:
                        matches.append(k)
            # matches = self.codes.map([j if c[i] == index_of_code_at_level\
            #       else None for j,c in codes])
            if len(matches) != 0:
                validity_mask = matches
        result = df.iloc[validity_mask]
        # Build new index
        from cudf import DataFrame
        out_index = DataFrame()
        for k in range(len(row_tuple), len(df.index.codes.columns)):
            out_index.add_column(df.index.names[k],
                                 df.index.codes[df.index.codes.columns[k]])
        if len(out_index.columns) == 1:
            print(result.index.codes)
            out_index = []
            for val in result.index.codes[result.index.codes.columns[len(result.index.codes.columns)-1]]:
                out_index.append(result.index.levels[
                        len(result.index.codes.columns)-1][val])
            out_index = StringIndex(out_index)
            out_index.name = result.index.names[len(result.index.names)-1]
            result.index = out_index
        return result

    def __len__(self):
        return len(self.codes[self.codes.columns[0]])

    def __eq__(self, other):
        return self.levels == other.levels and\
                self.codes == other.codes and\
                self.names == other.names

    def equals(self, other):
        return (self == other)._values.all()

    @property
    def is_contiguous(self):
        return True

    @property
    def size(self):
        return len(self.codes[0])

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
