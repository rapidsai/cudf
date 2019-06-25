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
        self._normalize()

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
        result_index_cols, result_cols = cpp_apply_groupby(self._key_columns,
                                                           self._value_columns,
                                                           agg)

        if self._sort or not self._as_index:
            result_key_names = self._key_names
            result_value_names = self._add_prefixes(self._value_names, agg)
            # concatenate results, sort and then split
            result = cudf.concat(
                [
                    dataframe_from_columns(result_index_cols, columns=result_key_names),
                    dataframe_from_columns(result_cols, columns=result_value_names)
                ],
                axis=1
            )
            result = result.sort_values(result_key_names)

            if not self._as_index:
                return result

            result_index_cols = columns_from_dataframe(result[result_key_names])
            result_cols = columns_from_dataframe(result[result_value_names])

        if self._as_index:
            result = dataframe_from_columns(
                result_cols,
                index=self._compute_result_index(result_index_cols),
                columns=self._compute_result_column_index(agg)
            )

        return result

    def _add_prefixes(self, names, prefixes):
        """
        Return a copy of ``names`` prefixed with ``prefixes``
        """
        prefixed_names = names.copy()
        if isinstance(prefixes, str):
            prefix = prefixes
            for i, col_name in enumerate(names):
                prefixed_names[i] = f"{prefix}_{col_name}"
        elif isinstance(prefixes, list):
            for i, col_name in enumerate(names):
                for prefix in prefixes:
                    prefixed_names[i] = f"{prefix}_{col_name}"
        else:
            for key, col_name in names.items():
                prefixed_names[key] = f"{prefix}_{col_name}"
        return prefixed_names

    def _normalize(self):
        """
        Normalizes the groupby object.
        Sets self._key_columns and self._value_columns.
        """
        if isinstance(self._by, str):
            self._key_names = [self._by]
        else:
            self._key_names = self._by
        self._value_names = list(set(self._df.columns) - set(self._key_names))
        self._key_columns = columns_from_dataframe(self._df[self._key_names])
        self._value_columns = columns_from_dataframe(self._df[self._value_names])

    def _compute_result_index(self, result_index_cols):
        """
        Computes the index of the result
        """
        if len(result_index_cols) == 1:
            return cudf.dataframe.index.as_index(result_index_cols[0],
                                                 name=self._key_names[0])
        else:
            return MultiIndex(source_data=dataframe_from_columns(result_index_cols,
                                                                 columns=self._key_names))

    def _compute_result_column_index(self, aggs):
        """
        Computes the column index of the result
        """
        if isinstance(aggs, str):
            return self._value_names
        elif isinstance(aggs, list):
            if len(aggs) == 1:
                return self._value_names
            else:
                return MultiIndex.from_tuples(zip(self._value_names, aggs))
        else:
            if all(isinstance(x, str) for x in aggs.values()):
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

