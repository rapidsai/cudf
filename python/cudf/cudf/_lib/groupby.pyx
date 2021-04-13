# Copyright (c) 2020, NVIDIA CORPORATION.

from collections import defaultdict
from pandas.core.groupby.groupby import DataError
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_string_dtype,
    is_list_dtype,
    is_interval_dtype,
    is_struct_dtype,
    is_decimal_dtype,
)

import numpy as np
import rmm

from libcpp.pair cimport pair
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from libcpp.vector cimport vector
from libcpp cimport bool

from cudf._lib.column cimport Column
from cudf._lib.table cimport Table
from cudf._lib.aggregation cimport Aggregation, make_aggregation

from cudf._lib.cpp.table.table cimport table, table_view
cimport cudf._lib.cpp.types as libcudf_types
cimport cudf._lib.cpp.groupby as libcudf_groupby


# The sets below define the possible aggregations that can be performed on
# different dtypes. These strings must be elements of the AggregationKind enum.
_CATEGORICAL_AGGS = {"COUNT", "SIZE", "NUNIQUE", "UNIQUE"}
_STRING_AGGS = {"COUNT", "SIZE", "MAX", "MIN", "NUNIQUE", "NTH", "COLLECT",
                "UNIQUE"}
_LIST_AGGS = {"COLLECT"}
_STRUCT_AGGS = set()
_INTERVAL_AGGS = set()
_DECIMAL_AGGS = {"COUNT", "SUM", "ARGMIN", "ARGMAX", "MIN", "MAX", "NUNIQUE",
                 "NTH", "COLLECT"}


cdef class GroupBy:
    cdef unique_ptr[libcudf_groupby.groupby] c_obj
    cdef dict __dict__

    def __cinit__(self, Table keys, bool dropna=True, *args, **kwargs):
        cdef libcudf_types.null_policy c_null_handling

        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef table_view keys_view = keys.view()

        with nogil:
            self.c_obj.reset(
                new libcudf_groupby.groupby(
                    keys_view,
                    c_null_handling,
                )
            )

    def __init__(self, Table keys, bool dropna=True):
        self.keys = keys
        self.dropna = dropna

    def groups(self, Table values):

        cdef table_view values_view = values.view()

        with nogil:
            c_groups = move(self.c_obj.get()[0].get_groups(values_view))

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
        cdef libcudf_groupby.aggregation_request c_agg_request
        cdef Column col
        cdef Aggregation agg_obj

        allow_empty = all(len(v) == 0 for v in aggregations.values())

        included_aggregations = defaultdict(list)
        for i, (col_name, aggs) in enumerate(aggregations.items()):
            col = values._data[col_name]
            dtype = col.dtype

            valid_aggregations = (
                _LIST_AGGS if is_list_dtype(dtype)
                else _STRING_AGGS if is_string_dtype(dtype)
                else _CATEGORICAL_AGGS if is_categorical_dtype(dtype)
                else _STRING_AGGS if is_struct_dtype(dtype)
                else _INTERVAL_AGGS if is_interval_dtype(dtype)
                else _DECIMAL_AGGS if is_decimal_dtype(dtype)
                else "ALL"
            )
            if (valid_aggregations is _DECIMAL_AGGS
                    and rmm._cuda.gpu.runtimeGetVersion() < 11000):
                raise RuntimeError(
                    "Decimal aggregations are only supported on CUDA >= 11 "
                    "due to an nvcc compiler bug."
                )

            c_agg_request = move(libcudf_groupby.aggregation_request())
            for agg in aggs:
                agg_obj = make_aggregation(agg)
                if (valid_aggregations == "ALL"
                        or agg_obj.kind in valid_aggregations):
                    included_aggregations[col_name].append(agg)
                    c_agg_request.aggregations.push_back(
                        move(agg_obj.c_obj)
                    )
            if not c_agg_request.aggregations.empty():
                c_agg_request.values = col.view()
                c_agg_requests.push_back(
                    move(c_agg_request)
                )

        if c_agg_requests.empty() and not allow_empty:
            raise DataError("All requested aggregations are unsupported.")

        cdef pair[
            unique_ptr[table],
            vector[libcudf_groupby.aggregation_result]
        ] c_result

        try:
            with nogil:
                c_result = move(
                    self.c_obj.get()[0].aggregate(
                        c_agg_requests
                    )
                )
        except RuntimeError as e:
            # TODO: remove this try..except after
            # https://github.com/rapidsai/cudf/issues/7611
            # is resolved
            if ("make_empty_column") in str(e):
                raise NotImplementedError(
                    "Aggregation not supported for empty columns"
                ) from e
            else:
                raise

        grouped_keys = Table.from_unique_ptr(
            move(c_result.first),
            column_names=self.keys._column_names
        )

        result_data = ColumnAccessor(multiindex=True)
        # Note: This loop relies on the included_aggregations dict being
        # insertion ordered to map results to requested aggregations by index.
        for i, col_name in enumerate(included_aggregations):
            for j, agg_name in enumerate(included_aggregations[col_name]):
                if callable(agg_name):
                    agg_name = agg_name.__name__
                result_data[(col_name, agg_name)] = (
                    Column.from_unique_ptr(move(c_result.second[i].results[j]))
                )

        return Table(data=result_data, index=grouped_keys)
