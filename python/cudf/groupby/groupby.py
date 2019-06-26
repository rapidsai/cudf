# Copyright (c) 2018, NVIDIA CORPORATION.

from numbers import Number
import collections
import itertools

import pandas as pd

import cudf
from cudf.dataframe.dataframe import DataFrame
from cudf.dataframe.series import Series
from cudf import MultiIndex
from cudf.bindings.nvtx import nvtx_range_pop

from cudf.bindings.groupby import apply_groupby as cpp_apply_groupby


def columns_from_dataframe(df):
    cols = [sr._column for sr in df._cols.values()]
    return cols


def dataframe_from_columns(cols, index_cols=None, index=None, columns=None):
    df = cudf.DataFrame(dict(zip(range(len(cols)), cols)), index=index)
    if columns is not None:
        df.columns = columns
    return df


class _Groupby(object):

    _NAMED_AGGS = ('sum', 'mean', 'min', 'max', 'count')

    def sum(self):
        return self._apply_aggregation("sum")

    def min(self):
        return self._apply_aggregation("min")

    def max(self):
        return self._apply_aggregation("max")

    def mean(self):
        return self._apply_aggregation("mean")

    def count(self):
        return self._apply_aggregation("count")

    def agg(self, func):
        return self._apply_aggregation(func)

class SeriesGroupBy(_Groupby):

    def __init__(self, sr, by, method="hash", level=None, sort=True):
        self._sr = sr
        self._by = by
        self._sort = sort
        self._normalize_keys()

    def _normalize_keys(self):
        """
        Normalizes the input key names and columns
        """
        if isinstance(self._by, (list, tuple)):
            self._key_names = []
            self._key_columns = []
            for by in self._by:
                name, col = self._key_from_by(by)
                self._key_names.append(name)
                self._key_columns.append(col)
        else:
            # grouping by a single label or Series
            name, col = self._key_from_by(self._by)
            self._key_names = [name]
            self._key_columns = [col]

    def _key_from_by(self, by):
        try:
            by = cudf.Series(by)
        except:
            raise ValueError("Cannot convert by argument to a Series")
        if len(by) != len(self._sr):
            raise NotImplementedError("cuDF does not support arbitrary series index lengths "
                                      "for groupby")
        key_name = by.name
        key_column = by._column
        return key_name, key_column

    def _apply_aggregation(self, agg):
        """
        Applies the aggregation function(s) ``agg`` on all columns
        """
        self._normalize_values(agg)

        out_key_columns, out_value_columns = _groupby_engine(
            self._key_columns,
            self._value_columns,
            self._aggs,
            self._sort
        )

        return self._construct_result(out_key_columns, out_value_columns)

    def _normalize_values(self, agg):
        unknown_agg_err = lambda agg: "Uknown aggregation function {}".format(agg)

        if isinstance(agg, str):
            if agg not in self._NAMED_AGGS:
                raise ValueError(unknown_arg_err(agg))
            self._aggs = agg
            self._value_columns = [self._sr._column]
            self._value_names = [self._sr.name]

        elif isinstance(agg, list):
            for agg_name in agg:
                if agg_name not in self._NAMED_AGGS:
                    raise ValueError(unknown_arg_err(agg))
            # repeat each element of self._value_columns len(agg) times,
            self._value_columns = [self._sr._column]*len(agg)
            self._value_names = agg
            self._aggs = agg
        else:
            raise ValueError("Invalid type for agg")

    def _construct_result(self, out_key_columns, out_value_columns):
        result = dataframe_from_columns(
                out_value_columns,
                columns=self._compute_result_column_index()
            )
        result.index = self._compute_result_index(out_key_columns, out_value_columns)

        if isinstance(self._aggs, str):
            result = result[result.columns[0]]
            result.name = self._value_names[0]
        return result

    def _compute_result_index(self, key_columns, value_columns):
        """
        Computes the index of the result
        """
        key_names = self._key_names
        if (len(key_columns)) == 1:
            return cudf.dataframe.index.as_index(key_columns[0],
                                                 name=key_names[0])
        else:
            empty_results = all([len(x)==0 for x in key_columns])
            if len(value_columns) == 0  and empty_results:
                return cudf.dataframe.index.GenericIndex(cudf.Series([], dtype='object'))
            return MultiIndex(source_data=dataframe_from_columns(key_columns,
                                                                 columns=key_names))

    def _compute_result_column_index(self):
        """
        Computes the column index of the result
        """
        value_names = self._value_names
        aggs = self._aggs
        if isinstance(aggs, str):
            return self._sr.name
        else:
            return self._aggs


class DataFrameGroupBy(_Groupby):

    def __init__(self, df, by, method="hash", as_index=True, level=None, sort=True):
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
        self._df = df
        self._by = by
        self._as_index = as_index
        self._sort = sort
        self._normalize_keys()

    def _normalize_keys(self):
        """
        Normalizes the input key names and columns
        """
        if isinstance(self._by, (list, tuple)):
            self._key_names = []
            self._key_columns = []
            for by in self._by:
                name, col = self._key_from_by(by)
                self._key_names.append(name)
                self._key_columns.append(col)
        else:
            # grouping by a single label or Series
            name, col = self._key_from_by(self._by)
            self._key_names = [name]
            self._key_columns = [col]

    def _key_from_by(self, by):
        if isinstance(by, str):
            key_name = by
            key_column = self._df[by]._column
        elif isinstance(by, cudf.Series):
            key_name = by.name
            key_column = by._column
        else:
            raise ValueError("Cannot group by object of type {}".format(
                type(by).__name__)
            )
        return key_name, key_column

    def _normalize_values(self, agg):
        """
        Normalizes the groupby object based on agg.
        """
        unknown_agg_err = lambda agg: "Uknown aggregation function {}".format(agg)

        self._value_names = [col for col in self._df.columns if col not in self._key_names]
        self._value_columns = columns_from_dataframe(self._df[self._value_names])

        if isinstance(agg, str):
            if agg not in self._NAMED_AGGS:
                raise ValueError(unknown_arg_err(agg))
            self._aggs = agg
        elif isinstance(agg, list):
            for agg_name in agg:
                if agg_name not in self._NAMED_AGGS:
                    raise ValueError(unknown_arg_err(agg))
            agg_list = agg * len(self._value_columns)
            # repeat each element of self._value_columns len(agg) times,
            # i.e., [A, B, C] -> [A, A, A..., B, B, B, ..., C, C, C, ...]
            self._value_columns = list(itertools.chain.from_iterable(
                len(agg)*[col] for col in self._value_columns
            ))
            self._value_names = list(itertools.chain.from_iterable(
                len(agg)*[col] for col in self._value_names
            ))
            self._aggs = agg_list
        elif isinstance(agg, dict):
            agg_list = []
            self._value_columns = []
            self._value_names = []
            for col_name, col_agg in agg.items():
                col = self._df[col_name]._column
                if isinstance(col_agg, str):
                    if col_agg not in self._NAMED_AGGS:
                        raise ValueError(unknown_agg_err(col_agg))
                    self._value_columns.append(col)
                    self._value_names.append(col_name)
                    agg_list.append(col_agg)
                elif isinstance(col_agg, list):
                    for col_sub_agg in col_agg:
                        if col_sub_agg not in self._NAMED_AGGS:
                            raise ValueError(unknown_agg_err(col_agg))
                    self._value_columns.extend([col]*len(col_agg))
                    self._value_names.extend([col_name]*len(col_agg))
                    agg_list.extend(col_agg)
            self._aggs = agg_list
        else:
            raise ValueError("Invalid type for agg")

    def _apply_aggregation(self, agg):
        """
        Applies the aggregation function(s) ``agg`` on all columns
        """
        self._normalize_values(agg)

        out_key_columns, out_value_columns = _groupby_engine(
            self._key_columns,
            self._value_columns,
            self._aggs,
            self._sort
        )

        return self._construct_result(out_key_columns, out_value_columns)


    def _construct_result(self, out_key_columns, out_value_columns):
        if self._as_index:
            result = dataframe_from_columns(
                out_value_columns,
                columns=self._compute_result_column_index()
            )
            result.index = self._compute_result_index(out_key_columns, out_value_columns)
        else:
            result = cudf.concat(
                [
                    dataframe_from_columns(out_key_columns, columns=self._key_names),
                    dataframe_from_columns(out_value_columns, columns=self._value_names)
                ],
                axis=1
            )
        return result

    def _compute_result_index(self, key_columns, value_columns):
        """
        Computes the index of the result
        """
        key_names = self._key_names
        if (len(key_columns)) == 1:
            return cudf.dataframe.index.as_index(key_columns[0],
                                                 name=key_names[0])
        else:
            empty_results = all([len(x)==0 for x in key_columns])
            if len(value_columns) == 0  and empty_results:
                return cudf.dataframe.index.GenericIndex(cudf.Series([], dtype='object'))
            return MultiIndex(source_data=dataframe_from_columns(key_columns,
                                                                 columns=key_names))

    def _compute_result_column_index(self):
        """
        Computes the column index of the result
        """
        value_names = self._value_names
        aggs = self._aggs
        if isinstance(aggs, str):
            return value_names
        else:
            if (
                    len(aggs) == 1
                    or len(set(value_names)) == len(value_names)
            ):
                return value_names
            else:
                return MultiIndex.from_tuples(zip(value_names, aggs))

    def __getitem__(self, arg):
        if isinstance(arg, str):
            return self.__getattr__(arg)
        else:
            arg = list(arg)
            import pdb
            pdb.set_trace()
            return self._df[arg].groupby(self._by,
                                         sort=self._sort)

    def __getattr__(self, key):
        if key not in self._df.columns:
            raise AttributeError("'DataFrameGroupBy' object has no attribute "
                                 "'{}'".format(key))
        by_list = []
        for by_name, by in zip(self._key_names, self._key_columns):
            by_list.append(cudf.Series(by, name=by_name))
        return self._df[key].groupby(by_list, sort=self._sort)


def _groupby_engine(key_columns, value_columns, aggs, sort):
    """
    Parameters
    ----------
    key_columns : list of Columns
    value_columns : list of Columns
    aggs : list of str
    sort : bool

    Returns
    -------
    out_key_columns : list of Columns
    out_value_columns : list of Columns
    """
    out_key_columns, out_value_columns = cpp_apply_groupby(
        key_columns,
        value_columns,
        aggs
    )

    if sort:
        key_names = ["key_"+str(i) for i in range(len(key_columns))]
        value_names = ["value_"+str(i) for i in range(len(value_columns))]
        value_names = _add_prefixes(value_names, aggs)

        # concatenate
        result = cudf.concat(
            [
                dataframe_from_columns(out_key_columns, columns=key_names),
                dataframe_from_columns(out_value_columns, columns=value_names)
            ],
            axis=1
        )

        # sort values
        result = result.sort_values(key_names)

        # split
        out_key_columns = columns_from_dataframe(result[key_names])
        out_value_columns = columns_from_dataframe(result[value_names])

    return out_key_columns, out_value_columns

def _add_prefixes(names, prefixes):
    """
    Return a copy of ``names`` prefixed with ``prefixes``
    """
    prefixed_names = names.copy()
    if isinstance(prefixes, str):
        prefix = prefixes
        for i, col_name in enumerate(names):
            prefixed_names[i] = f"{prefix}_{col_name}"
    else:
        for i, (prefix, col_name) in enumerate(zip(prefixes, names)):
            prefixed_names[i] = f"{prefix}_{col_name}"
    return prefixed_names

