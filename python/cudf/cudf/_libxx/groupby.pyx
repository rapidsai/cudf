from collections import defaultdict

import numpy as np

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.column cimport Column
from cudf._libxx.table cimport Table
from cudf._libxx.move cimport move
from cudf._libxx.aggregation cimport make_aggregation, Aggregation

from cudf._libxx.cpp.table.table cimport table
cimport cudf._libxx.cpp.groupby as libcudf_groupby
cimport cudf._libxx.cpp.aggregation as libcudf_aggregation


cdef class GroupBy:
    cdef unique_ptr[libcudf_groupby.groupby] c_obj
    cdef dict __dict__

    def __cinit__(self, Table keys, *args, **kwargs):
        """
        keys : a GroupByKeys object
        """
        self.c_obj.reset(
            new libcudf_groupby.groupby(
                keys.view()
            )
        )

    def __init__(self, keys):
        self.keys = keys

    def groups(self, Table values):
        c_groups = move(self.c_obj.get()[0].get_groups(values.view()))
        c_grouped_keys = move(c_groups.keys)
        c_grouped_values = move(c_groups.values)
        c_group_offsets = c_groups.offsets

        grouped_keys = Table.from_unique_ptr(
            move(c_grouped_keys),
            column_names=range(c_grouped_keys.get()[0].num_columns())
        )
        grouped_values = Table.from_unique_ptr(
            move(c_grouped_values),
            index_names=values._index_names,
            column_names=values._column_names
        )
        return grouped_keys, grouped_values, c_group_offsets

    def aggregate(self, Table values, aggregations):
        """
        Parameters
        ----------
        values : Table
        aggregations
            A dict mapping column names in `Table` to a list of aggregations
            to perform on that column

            Each aggregation may be specified as:
            - a string (e.g., "max")
            - a lambda/function

        Returns
        -------
        Table of aggregated values
        """
        from cudf.core.column_accessor import ColumnAccessor
        cdef vector[libcudf_groupby.aggregation_request] c_agg_requests
        cdef Column col

        aggregations = _drop_unsupported_aggs(values, aggregations)

        for i, (col_name, aggs) in enumerate(aggregations.items()):
            col = values._data[col_name]
            c_agg_requests.push_back(
                libcudf_groupby.aggregation_request()
            )
            c_agg_requests[i].values = col.view()
            for agg in aggs:
                c_agg_requests[i].aggregations.push_back(
                    move(make_aggregation(agg))
                )

        cdef pair[
            unique_ptr[table],
            vector[libcudf_groupby.aggregation_result]
        ] c_result

        with nogil:
            c_result = move(
                self.c_obj.get()[0].aggregate(
                    c_agg_requests
                )
            )

        grouped_keys = Table.from_unique_ptr(
            move(c_result.first),
            column_names=self.keys._column_names
        )

        result_data = ColumnAccessor(multiindex=True)
        for i, col_name in enumerate(aggregations):
            for j, agg_name in enumerate(aggregations[col_name]):
                if callable(agg_name):
                    agg_name = agg_name.__name__
                result_data[(col_name, agg_name)] = (
                    Column.from_unique_ptr(move(c_result.second[i].results[j]))
                )

        result = Table(data=result_data, index=grouped_keys)
        return result

_STRING_AGGS = [
    "count_valid",
    "count_all",
    "max",
    "min",
    "nth_element",
    "nunique"
]

def _drop_unsupported_aggs(Table values, aggs):
    """
    Normalize aggs to a dictionary mapping
    column names to a list of _Aggregation objects.

    Drop any aggregations that are not supported.
    """
    from cudf.utils.dtypes import (
        is_categorical_dtype,
        is_string_dtype
    )
    result = aggs.copy()

    for col_name in aggs:
        for i, agg_name in enumerate(aggs[col_name]):
            if (
                    is_string_dtype(values._data[col_name].dtype)
                    or is_categorical_dtype(values._data[col_name].dtype)
            ):
                if Aggregation(agg_name).kind not in _STRING_AGGS:
                    del result[col_name][i]

    return result
