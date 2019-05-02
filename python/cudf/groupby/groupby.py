# Copyright (c) 2018, NVIDIA CORPORATION.

import numpy as np

from numbers import Number

from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf.bindings.groupby import (
    agg as cpp_agg,
    _apply_basic_agg as _cpp_apply_basic_agg
)


class SeriesGroupBy(object):
    """Wraps DataFrameGroupby with special attr methods
    """
    def __init__(self, source_series, group_series, level=None, sort=False):
        self.source_series = source_series
        self.group_series = group_series
        self.level = level
        self.sort = sort

    def __getattr__(self, attr):
        df = DataFrame()
        df['x'] = self.source_series
        if self.level is not None:
            df['y'] = self.source_series.index
        else:
            df['y'] = self.group_series
        groupby = df.groupby('y', level=self.level, sort=self.sort)
        result_df = getattr(groupby, attr)()

        def get_result():
            result_series = result_df['x']
            result_series.name = None
            idx = result_df.index
            idx.name = None
            result_series.set_index(idx)
            return result_series
        return get_result

    def agg(self, agg_types):
        df = DataFrame()
        df['x'] = self.source_series
        if self.level is not None:
            df['y'] = self.source_series.index
        else:
            df['y'] = self.group_series
        groupby = df.groupby('y').agg(agg_types)
        idx = groupby.index
        idx.name = None
        groupby.set_index(idx)
        return groupby


class Groupby(object):
    """Groupby object returned by cudf.DataFrame.groupby().
    """

    _LEVEL_0_INDEX_NAME = 'cudf_groupby_level_index'

    def __init__(self, df, by, method="hash", as_index=True, level=None):
        """
        Parameters
        ----------
        df : DataFrame
        by : str, list
            - str
                The column name to group on.
            - list
                List of *str* of the column names to group on.
        method : str, optional
            A string indicating the libgdf method to use to perform the
            group by. Valid values are "hash".
        """
        self.level = None
        self._original_index_name = None
        self._df = df
        if isinstance(by, Series):
            if len(by) != len(self._df.index):
                raise NotImplementedError("CUDF doesn't support series groupby"
                                          "with indices of arbitrary length")
            self.level = 0
            self._df[self._LEVEL_0_INDEX_NAME] = by
            self._original_index_name = self._df.index.name
            self._by = [self._LEVEL_0_INDEX_NAME]
        elif level == 0:
            self.level = level
            self._df[self._LEVEL_0_INDEX_NAME] = self._df.index
            self._original_index_name = self._df.index.name
            self._by = [self._LEVEL_0_INDEX_NAME]
        elif level and level > 0:
            raise NotImplementedError('MultiIndex not supported yet in cudf')
        else:
            self._by = [by] if isinstance(by, (str, Number)) else list(by)
        self._val_columns = [idx for idx in self._df.columns
                             if idx not in self._by]
        self._as_index = as_index
        if (method == "hash"):
            self._method = method
        else:
            msg = "Method {!r} is not a supported group by method"
            raise NotImplementedError(msg.format(method))

    def _apply_basic_agg(self, agg_type, sort_results=False):
        """
        Parameters
        ----------
        agg_type : str
            The aggregation function to run.
        """
        return _cpp_apply_basic_agg(self, agg_type, sort_results=sort_results)

    def apply_multiindex_or_single_index(self, result):
        if len(self._by) == 1:
            from cudf.dataframe import index
            idx = index.as_index(result[self._by[0]])
            idx.name = self._by[0]
            result = result.drop(idx.name)
            if idx.name == self._LEVEL_0_INDEX_NAME:
                idx.name = None
            result = result.set_index(idx)
            return result
        else:
            levels = []
            codes = DataFrame()
            names = []
            # Note: This is an O(N^2) solution using gpu masking
            # to compute new codes for the MultiIndex. There may be
            # a faster solution that could be executed on gpu at the same
            # time the groupby is calculated.
            for by in self._by:
                level = result[by].unique()
                code = result[by]
                for idx, value in enumerate(level):
                    level_mask = code == value
                    code = code.masked_assign(idx, level_mask)
                levels.append(level)
                codes[by] = code
                names.append(by)
            from cudf import MultiIndex
            multi_index = MultiIndex(levels=levels,
                                     codes=codes,
                                     names=names)
            final_result = DataFrame()
            for col in result.columns:
                if col not in self._by:
                    final_result[col] = result[col]
            return final_result.set_index(multi_index)

    def apply_multicolumn(self, result, aggs):
        levels = []
        codes = []
        levels.append(self._val_columns)
        levels.append(aggs)
        codes.append(list(np.zeros(len(aggs), dtype='int64')))
        codes.append(list(range(len(aggs))))
        from cudf import MultiIndex
        result.columns = MultiIndex(levels, codes)
        return result

    def apply_multicolumn_mapped(self, result, aggs):
        if len(set(aggs.keys())) == len(aggs.keys()) and\
                isinstance(aggs[list(aggs.keys())[0]], (str, Number)):
            result.columns = aggs.keys()
        else:
            tuples = []
            for k in aggs.keys():
                for v in aggs[k]:
                    tuples.append((k, v))
            from cudf import MultiIndex
            multiindex = MultiIndex.from_tuples(tuples)
            result.columns = multiindex
        return result

    def __getitem__(self, arg):
        if isinstance(arg, (str, Number)):
            if arg not in self._val_columns:
                raise KeyError("Column not found: " + str(arg))
        else:
            for val in arg:
                if val not in self._val_columns:
                    raise KeyError("Column not found: " + str(val))
        result = self.copy()
        result._val_columns = arg
        return result

    def copy(self, deep=True):
        df = self._df.copy(deep) if deep else self._df
        result = Groupby(df, self._by)
        result._method = self._method
        result._val_columns = self._val_columns
        result.level = self.level
        result._original_index_name = self._original_index_name
        return result

    def __getattr__(self, key):
        if key != '_val_columns' and key in self._val_columns:
            return self[key]
        raise AttributeError("'Groupby' object has no attribute %r" % key)

    def min(self, sort=True):
        return self._apply_basic_agg("min", sort)

    def max(self, sort=True):
        return self._apply_basic_agg("max", sort)

    def count(self, sort=True):
        return self._apply_basic_agg("count", sort)

    def sum(self, sort=True):
        return self._apply_basic_agg("sum", sort)

    def mean(self, sort=True):
        return self._apply_basic_agg("mean", sort)

    def agg(self, args):
        """ Invoke aggregation functions on the groups.

        Parameters
        ----------
        args : dict, list, str, callable
            - str
                The aggregate function name.
            - list
                List of *str* of the aggregate function.
            - dict
                key-value pairs of source column name and list of
                aggregate functions as *str*.

        Returns
        -------
        result : DataFrame

        Notes
        -----
        Since multi-indexes aren't supported aggregation results are returned
        in columns using the naming scheme of `aggregation_columnname`.
        """
        return cpp_agg(self, args)
