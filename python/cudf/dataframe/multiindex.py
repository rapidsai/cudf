# Copyright (c) 2019, NVIDIA CORPORATION.

import pandas as pd
import numpy as np

from cudf.dataframe import columnops
from cudf.comm.serialize import register_distributed_serializer
import cudf.dataframe.index as index

class MultiIndex(index.Index):
    """A multi-level or hierarchical index.

    Provides N-Dimensional indexing into Series and DataFrame objects.
    
    Properties
    ---
    levels: Labels for each category in the index hierarchy.
    codes: Assignment of individual items into the categories of the hierarchy.
    names: Name for each level
    """

    def __init__(self, levels, codes, names=None):
        self._validate_levels_and_codes(levels, codes)
        self.levels = levels
        from cudf import DataFrame
        self.codes = DataFrame()
        if names is None:
            column_names = list(range(len(levels)))
        else:
            column_names = names
        for index, code in enumerate(codes):
            self.codes.add_column(column_names[index],
                    columnops.as_column(np.array(code)))
        self.names = names

    def _validate_levels_and_codes(self, levels, codes):
        levels = np.array(levels)
        codes = np.array(codes)
        if len(levels) != len(codes):
            raise ValueError('MultiIndex has unequal number of levels and codes and is inconsistent!')
        code_length = len(codes[0])
        for index, code in enumerate(codes):
            if code_length != len(code):
                raise ValueError('MultiIndex length of codes does not match and is inconsistent!')
        for index, code in enumerate(codes):
            if code.max() > len(levels[index])-1:
                raise ValueError('MultiIndex code %d contains value %d larger than maximum level size at this position')

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
            # return a new MultiIndex with the sliced codes, same levels and names
            None
        elif isinstance(index, int):
            # return a tuple of the labels of the item defined by the set of codes
            # at the ith positionjA
            None
        elif isinstance(index, (list, np.ndarray)):
            # return a new MultiIndex constructed similarly to the slice case
            None
        if isinstance(index, (DeviceNDArray)):
            return self.take(index)
        else:
            raise IndexError('only integers, slices (`:`), ellipsis (`...`),'
            'numpy.newaxis (`None`) and integer or boolean arrays are valid indices')

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
