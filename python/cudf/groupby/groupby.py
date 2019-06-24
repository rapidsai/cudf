# Copyright (c) 2018, NVIDIA CORPORATION.

from numbers import Number

from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf import MultiIndex

from cudf.bindings.groupby import (
    agg as cpp_agg,
    apply_basic_agg as cpp_apply_basic_agg
)


_LEVEL_0_INDEX_NAME = 'cudf_groupby_level_index'
_LEVEL_0_DATA_NAME = 'cudf_groupby_data_name'


class SeriesGroupBy(object):
    """Wraps DataFrameGroupby with special attr methods
    """
    def __init__(self, source_series, group_keys, level=None, sort=False,
                 by=None):
        self._by = by
        self.source_series = source_series
        self.source_name = _LEVEL_0_DATA_NAME
        if self.source_series is not None and\
                self.source_series.name is not None:
            self.source_name = source_series.name
        self.group_keys = group_keys
        self.group_name = _LEVEL_0_INDEX_NAME
        if self.group_keys is not None and\
                self.group_keys.name is not None:
            self.group_name = group_keys.name
        self.level = level
        self.sort = sort

    def __getattr__(self, attr):
        df = DataFrame()
        by = []
        if self.level is not None:
            if isinstance(self.source_series.index, MultiIndex):
                # Add index columns specified by multiindex into _df
                # Record the index column names for the groupby
                for col in self.source_series.index.codes:
                    df[self.group_name + col] = self.source_series.index.codes[
                            col]
                    by.append(self.group_name + col)
        else:
            if isinstance(self.group_keys, Series):
                df[self.group_name] = self.group_keys
                by = self.group_name
            else:
                df = self.group_keys
                by = self._by
        df[self.source_name] = self.source_series
        groupby = df.groupby(by,
                             level=self.level,
                             sort=self.sort)
        if attr in ['sum', 'min', 'max', 'mean', 'count', 'agg']:
            result_df = getattr(groupby, attr)()
        else:
            return getattr(groupby, attr)

        def get_result():
            result_series = result_df[self.source_name]
            result_series.name = self.source_name if self.source_name !=\
                _LEVEL_0_DATA_NAME else None
            if len(result_df) == 0 and self._by is not None:
                empties = [[] for x in range(len(self._by))]
                mi = MultiIndex(empties, empties, names=self._by)
                result_series = result_series.set_index(mi)
            else:
                idx = result_df.index
                if self.group_name == _LEVEL_0_INDEX_NAME:
                    idx.name = None
                result_series = result_series.set_index(idx)
            return result_series
        return get_result

    def agg(self, agg_types):
        df = DataFrame()
        by = []
        if self.level is not None:
            if isinstance(self.source_series.index, MultiIndex):
                # Add index columns specified by multiindex into _df
                # Record the index column names for the groupby
                for col in self.source_series.index.codes:
                    df[self.group_name + col] = self.source_series.index.codes[
                            col]
                    by.append(self.group_name + col)
        else:
            if isinstance(self.group_keys, Series):
                df[self.group_name] = self.group_keys
                by = self.group_name
            else:
                df = self.group_keys
                by = self._by
        df[self.source_name] = self.source_series
        groupby = df.groupby(by).agg(agg_types)
        idx = groupby.index
        if len(groupby.columns) == 1:
            result = groupby[self.source_name]
            result.name = self.source_series.name
            idx.name = None
            result = result.set_index(idx)
        else:
            idx.name = self.group_name
            result = groupby.set_index(idx)
        if len(result) == 0 and self._by is not None:
            empties = [[] for x in range(len(self._by))]
            mi = MultiIndex(empties, empties, names=self._by)
            result = result.set_index(mi)
        return result


class Groupby(object):
    """Groupby object returned by cudf.DataFrame.groupby().
    """

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
        self.level = level
        self._original_index_name = None
        self._val_columns = []
        self._df = df.copy(deep=False)
        self._as_index = as_index
        if len(df) == 0:  # empty case
            if by is None or isinstance(by, str):
                self._by = [by]
            else:
                self._by = list(by)
            self._df = df
            self._val_columns = []
            if by is not None:
                by = [by] if isinstance(by, str) else list(by)
            for col in self._df.columns:
                if by is None or col not in by:
                    self._val_columns.append(col)
            self._method = method
            return
        if isinstance(by, Series):
            if len(by) != len(self._df.index):
                raise NotImplementedError("CUDF doesn't support series groupby"
                                          "with indices of arbitrary length")
            self.level = 0
            self._df[_LEVEL_0_INDEX_NAME] = by
            self._original_index_name = self._df.index.name
            self._by = [_LEVEL_0_INDEX_NAME]
        elif level is not None:
            if level == 0:
                self._df[_LEVEL_0_INDEX_NAME] = self._df.index
                self._original_index_name = self._df.index.name
                self._by = [_LEVEL_0_INDEX_NAME]
            else:
                level = [level] if isinstance(
                        level, (str, Number)) else list(level)
                self._by = []
                # guard against missing MI names
                if self._df.index.names is None or sum(
                        x is None for x in self._df.index.names) > 1:
                    self._df.index.names = list(
                            range(len(self._df.index._source_data.columns)))
                for which_level in level:
                    # find the index of the level in the MultiIndex
                    if isinstance(which_level, str):
                        for idx, name in enumerate(self._df.index.names):
                            if name == which_level:
                                which_level = idx
                                break
                    try:
                        level_values = self._df.index._source_data[
                                self._df.index._source_data.columns[
                                    which_level]]
                    except IndexError:
                        raise IndexError("Too many levels: Index has only %d levels, not %d" % (len(self._df.index._source_data.columns), which_level+1))  # noqa: E501
                    self._df[self._df.index.names[which_level]] = level_values
                    self._by.append(self._df.index.names[which_level])
        else:
            self._by = [by] if isinstance(by, (str, Number)) else list(by)
        if isinstance(self._by[0], (str, Number)):
            # by is a list of column names or numerals
            # The base case!
            # Everything else in __init__ handles more complicated
            # configurations of "by"
            self._val_columns = [idx for idx in self._df.columns
                                 if idx not in self._by]
        else:
            # by is a list of objects - lists, or Series
            self._val_columns = self._df.columns
            by = self._by
            self._by = []
            for idx, each_by in enumerate(by):
                self._df['cudfvalcol+' + each_by.name] = each_by
                self._by.append('cudfvalcol+' + each_by.name)
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
        agg_groupby = self.copy(deep=False)
        if agg_type in ['mean', 'sum']:
            agg_groupby._df = agg_groupby._df._get_numeric_data()
            agg_groupby._val_columns = []
            columns = [self._val_columns] if isinstance(self._val_columns, (
                  str, Number)) else list(self._val_columns)
            for val in columns:
                if val in agg_groupby._df.columns:
                    agg_groupby._val_columns.append(val)
            # _get_numeric_data might have removed the by column
            if isinstance(self._by, (str, Number)):
                agg_groupby._df[self._by] = self._df[self._by]
            else:
                for by in self._by:
                    agg_groupby._df._cols[by] = self._df._cols[by]
        return cpp_apply_basic_agg(agg_groupby, agg_type,
                                   sort_results=sort_results)

    def apply_multiindex_or_single_index(self, result):
        if len(result) == 0:
            final_result = DataFrame()
            for col in result.columns:
                if col not in self._by:
                    final_result[col] = result[col]
            if len(self._by) == 1 or len(final_result.columns) == 0:
                if len(self._by) == 1:
                    dtype = self._df[self._by[0]]
                else:
                    dtype = 'object'
                name = self._by[0] if len(self._by) == 1 else None
                from cudf.dataframe.index import GenericIndex
                index = GenericIndex(Series([], dtype=dtype))
                index.name = name
                final_result.index = index
            else:
                mi = MultiIndex(source_data=result[self._by])
                mi.names = self._by
                final_result.index = mi
            return final_result
        if len(self._by) == 1:
            from cudf.dataframe import index
            idx = index.as_index(result[self._by[0]])
            name = self._by[0]
            if isinstance(name, str):
                name = self._by[0].split('+')
                if name[0] == 'cudfvalcol':
                    idx.name = name[1]
                else:
                    idx.name = name[0]
                result = result.drop(self._by[0])
            for col in result.columns:
                if isinstance(col, str):
                    colnames = col.split('+')
                    if colnames[0] == 'cudfvalcol':
                        result[colnames[1]] = result[col]
                        result = result.drop(col)
            if idx.name == _LEVEL_0_INDEX_NAME:
                idx.name = self._original_index_name
            result = result.set_index(idx)
            return result
        else:
            for col in result.columns:
                if isinstance(col, str):
                    colnames = col.split('+')
                    if colnames[0] == 'cudfvalcol':
                        result[colnames[1]] = result[col]
                        result = result.drop(col)
            new_by = []
            for by in self._by:
                if isinstance(col, str):
                    splitby = by.split('+')
                    if splitby[0] == 'cudfvalcol':
                        new_by.append(splitby[1])
                    else:
                        new_by.append(splitby[0])
                else:
                    new_by.append(by)
            self._by = new_by
            multi_index = MultiIndex(source_data=result[self._by])
            final_result = DataFrame()
            for col in result.columns:
                if col not in self._by:
                    final_result[col] = result[col]
            if len(final_result.columns) > 0:
                return final_result.set_index(multi_index)
            else:
                return result.set_index(multi_index)

    def apply_multicolumn(self, result, aggs):
        # multicolumn only applies with multiple aggs and multiple groupby keys
        if len(aggs) == 1 or len(self._by) == 1:
            # unprefix columns
            new_cols = []
            for c in result.columns:
                if len(self._by) == 1 and len(result) != 0:
                    new_col = c.split('_')[0]  # sum_z-> (sum, z)
                else:
                    new_col = c.split('_')[1]  # sum_z-> (sum, z)
                new_cols.append(new_col)
            result.columns = new_cols
        else:
            # reorder our columns to match pandas
            if len(self._val_columns) > 1:
                col_dfs = [DataFrame() for col in self._val_columns]
                for agg in aggs:
                    for idx, col in enumerate(self._val_columns):
                        col_dfs[idx][agg + '_' + col] = result[agg + '_' + col]
                idx = result.index
                result = DataFrame(index=idx)
                for idx, col in enumerate(self._val_columns):
                    for agg in aggs:
                        result[agg + '_' + col] = col_dfs[idx][agg + '_' + col]

            levels = []
            codes = []
            levels.append(self._val_columns)
            levels.append(aggs)

            # if the values columns have length == 1, codes is a nested list of
            # zeros equal to the size of aggs (sum, min, mean, etc.)
            # if the values columns are length>1, codes will monotonically
            # increase by 1 for every n values where n is the number of aggs
            # [['x,', 'z'], ['sum', 'min']]
            # codes == [[0, 1], [0, 1]]
            first_codes = [len(aggs) * [d*1] for d in range(len(
                    self._val_columns))]
            codes.append([item for sublist in first_codes for item in sublist])
            codes.append(len(self._val_columns) * [0, 1])
            result.columns = MultiIndex(levels, codes)
        return result

    def apply_multicolumn_mapped(self, result, aggs):
        # if all of the aggregations in the mapping set are only
        # length 1, we can assign the columns directly as keys.
        can_map_directly = True
        for values in aggs.values():
            value = [values] if isinstance(values, str) else list(values)
            if len(value) != 1:
                can_map_directly = False
                break
        if can_map_directly:
            result.columns = aggs.keys()
        else:
            tuples = []
            for k in aggs.keys():
                value = [aggs[k]] if isinstance(aggs[k], str) else list(
                        aggs[k])
                for v in value:
                    tuples.append((k, v))
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
        result = self.copy(deep=False)
        result._df = DataFrame()
        if isinstance(self._by, (str, Number)):
            result._df[self._by] = self._df[self._by]
        else:
            for by in self._by:
                result._df[by] = self._df[by]
        result._val_columns = arg
        if isinstance(arg, (str, Number)):
            result._df[arg] = self._df[arg]
        else:
            for a in arg:
                result._df[a] = self._df[a]
        if isinstance(result._val_columns,
                      (str, Number)):
            new_by = [result._by] if isinstance(result._by, (str, Number))\
                else list(result._by)
            new_val_columns = [result._val_columns] if\
                isinstance(result._val_columns, (str, Number))\
                else list(result._val_columns)
            new_val_series = result._df[new_val_columns[0]]
            new_val_series.name = new_val_columns[0]
            new_by_keys = result._df[new_by]
            new_by_keys.name = new_by[0]
            return SeriesGroupBy(new_val_series,
                                 new_by_keys, by=result._by)
        return result

    def copy(self, deep=True):
        df = self._df.copy(deep) if deep else self._df
        result = Groupby(df,
                         self._by,
                         method=self._method,
                         as_index=self._as_index,
                         level=self.level)
        result._original_index_name = self._original_index_name
        return result

    def deepcopy(self):
        return self.copy(deep=True)

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
        """
        agg_groupby = self.copy(deep=False)
        return cpp_agg(agg_groupby, args)
