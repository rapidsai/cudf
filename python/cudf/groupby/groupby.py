# Copyright (c) 2018, NVIDIA CORPORATION.

from numbers import Number
import collections
import itertools

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


class DataFrameGroupby(object):

    _NAMED_AGGS = ('sum', 'mean', 'min', 'max', 'count')

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

    def _apply_aggregation(self, agg):
        """
        Applies the aggregation function(s) ``agg`` on all columns
        """
        agg = self._normalize_agg(agg)
        out_key_columns, out_value_columns = _groupby_engine(
            self._key_columns,
            self._value_columns,
            agg,
            self._sort
        )

        if self._as_index:
            result = dataframe_from_columns(
                out_value_columns,
                index=self._compute_result_index(out_key_columns),
                columns=self._compute_result_column_index(self._value_names, agg)
            )
        else:
            result = cudf.concat(
                [
                    dataframe_from_columns(out_key_columns, columns=self._key_names),
                    dataframe_from_columns(out_value_columns, columns=self._value_names)
                ],
                axis=1
            )
        return result

    def _compute_result_index(self, result_key_columns):
        """
        Computes the index of the result
        """
        if (len(result_key_columns)) == 1:
            return cudf.dataframe.index.as_index(result_key_columns[0],
                                                 name=self._key_names[0])
        else:
            return MultiIndex(source_data=dataframe_from_columns(result_key_columns,
                                                                 columns=self._key_names))

    def _compute_result_column_index(self, result_column_names, aggs):
        """
        Computes the column index of the result
        """
        if isinstance(aggs, str):
            return self._value_names
        else:
            if (
                    len(aggs) == 1
                    or len(set(result_column_names)) == len(result_column_names)
            ):
                return self._value_names
            else:
                return MultiIndex.from_tuples(zip(self._value_names, aggs))


    def _normalize_agg(self, agg):
        """
        Normalizes the groupby object based on agg.

        1. Sets self._value_columns and self._value_names based on agg
        2. Returns ``agg`` as either a single string or a list of strings
           corresponding to the aggregation that needs to be
           performed on each of ``self._value_columns``.

        If agg is returned as a list, its length is equal
        to ``self._value_columns``.
        """
        unknown_agg_err = lambda agg: "Uknown aggregation function {}".format(agg)

        if isinstance(self._by, str):
            self._key_names = [self._by]
        else:
            self._key_names = self._by
        self._key_columns = columns_from_dataframe(self._df[self._key_names])

        self._value_names = [col for col in self._df.columns if col not in self._key_names]
        self._value_columns = columns_from_dataframe(self._df[self._value_names])

        if isinstance(agg, str):
            if agg not in self._NAMED_AGGS:
                raise ValueError(unknown_arg_err(agg))
            return agg
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
            return agg_list
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
            return agg_list
        else:
            raise ValueError("Invalid type for agg")


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

