# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from collections import defaultdict

import numpy as np
from pandas.core.groupby.groupby import DataError

import rmm

from cudf.api.types import (
    is_categorical_dtype,
    is_decimal_dtype,
    is_interval_dtype,
    is_list_dtype,
    is_string_dtype,
    is_struct_dtype,
)

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

import cudf

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.utils cimport (
    columns_from_unique_ptr,
    data_from_unique_ptr,
    table_view_from_columns,
    table_view_from_table,
)

from cudf._lib.scalar import as_device_scalar

cimport cudf._lib.cpp.groupby as libcudf_groupby
cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.aggregation cimport (
    GroupbyAggregation,
    GroupbyScanAggregation,
    make_groupby_aggregation,
    make_groupby_scan_aggregation,
)
from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.libcpp.functional cimport reference_wrapper
from cudf._lib.cpp.replace cimport replace_policy
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.table.table cimport table, table_view
from cudf._lib.cpp.types cimport size_type
from cudf._lib.cpp.utilities.host_span cimport host_span

# The sets below define the possible aggregations that can be performed on
# different dtypes. These strings must be elements of the AggregationKind enum.
_CATEGORICAL_AGGS = {"COUNT", "SIZE", "NUNIQUE", "UNIQUE"}
_STRING_AGGS = {"COUNT", "SIZE", "MAX", "MIN", "NUNIQUE", "NTH", "COLLECT",
                "UNIQUE"}
_LIST_AGGS = {"COLLECT"}
_STRUCT_AGGS = {"CORRELATION", "COVARIANCE"}
_INTERVAL_AGGS = set()
_DECIMAL_AGGS = {"COUNT", "SUM", "ARGMIN", "ARGMAX", "MIN", "MAX", "NUNIQUE",
                 "NTH", "COLLECT"}

# workaround for https://github.com/cython/cython/issues/3885
ctypedef const scalar constscalar


cdef _agg_result_from_columns(
    vector[libcudf_groupby.aggregation_result]& c_result_columns,
    set column_included,
    int n_input_columns
):
    """Construct the list of result columns from libcudf result. The result
    contains the same number of lists as the number of input columns. Result
    for an input column that has no applicable aggregations is an empty list.
    """
    cdef:
        int i
        int j
        int result_index = 0
        vector[unique_ptr[column]]* c_result
    result_columns = []
    for i in range(n_input_columns):
        if i in column_included:
            c_result = &c_result_columns[result_index].results
            result_columns.append([
                Column.from_unique_ptr(move(c_result[0][j]))
                for j in range(c_result[0].size())
            ])
            result_index += 1
        else:
            result_columns.append([])
    return result_columns

cdef class GroupBy:
    cdef unique_ptr[libcudf_groupby.groupby] c_obj
    cdef dict __dict__

    def __cinit__(self, list keys, bool dropna=True, *args, **kwargs):
        cdef libcudf_types.null_policy c_null_handling

        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef table_view keys_view = table_view_from_columns(keys)

        with nogil:
            self.c_obj.reset(
                new libcudf_groupby.groupby(
                    keys_view,
                    c_null_handling,
                )
            )

    def __init__(self, list keys, bool dropna=True):
        self.keys = keys
        self.dropna = dropna

    def groups(self, list values):
        cdef table_view values_view = table_view_from_columns(values)

        with nogil:
            c_groups = move(self.c_obj.get()[0].get_groups(values_view))

        grouped_key_cols = columns_from_unique_ptr(move(c_groups.keys))
        grouped_value_cols = columns_from_unique_ptr(move(c_groups.values))
        return grouped_key_cols, grouped_value_cols, c_groups.offsets

    def aggregate_internal(self, values, aggregations):
        """`values` is a list of columns and `aggregations` is a list of list
        of aggregations. `aggregations[i]` is a list of aggregations for
        `values[i]`. Returns a tuple containing 1) list of list of aggregation
        results, 2) a list of grouped keys, and 3) a list of list of
        aggregations performed.
        """
        cdef vector[libcudf_groupby.aggregation_request] c_agg_requests
        cdef libcudf_groupby.aggregation_request c_agg_request
        cdef Column col
        cdef GroupbyAggregation agg_obj

        cdef pair[
            unique_ptr[table],
            vector[libcudf_groupby.aggregation_result]
        ] c_result

        allow_empty = all(len(v) == 0 for v in aggregations)

        included_aggregations = []
        column_included = set()
        for i, (col, aggs) in enumerate(zip(values, aggregations)):
            dtype = col.dtype

            valid_aggregations = (
                _LIST_AGGS if is_list_dtype(dtype)
                else _STRING_AGGS if is_string_dtype(dtype)
                else _CATEGORICAL_AGGS if is_categorical_dtype(dtype)
                else _STRUCT_AGGS if is_struct_dtype(dtype)
                else _INTERVAL_AGGS if is_interval_dtype(dtype)
                else _DECIMAL_AGGS if is_decimal_dtype(dtype)
                else "ALL"
            )
            included_aggregations_i = []

            c_agg_request = move(libcudf_groupby.aggregation_request())
            for agg in aggs:
                agg_obj = make_groupby_aggregation(agg)
                if (valid_aggregations == "ALL"
                        or agg_obj.kind in valid_aggregations):
                    included_aggregations_i.append(agg)
                    c_agg_request.aggregations.push_back(
                        move(agg_obj.c_obj)
                    )
            included_aggregations.append(included_aggregations_i)
            if not c_agg_request.aggregations.empty():
                c_agg_request.values = col.view()
                c_agg_requests.push_back(
                    move(c_agg_request)
                )
                column_included.add(i)
        if c_agg_requests.empty() and not allow_empty:
            raise DataError("All requested aggregations are unsupported.")

        with nogil:
            c_result = move(
                self.c_obj.get()[0].aggregate(
                    c_agg_requests
                )
            )

        grouped_keys = columns_from_unique_ptr(
            move(c_result.first)
        )

        result_columns = _agg_result_from_columns(
            c_result.second, column_included, len(values)
        )

        return result_columns, grouped_keys, included_aggregations

    def scan_internal(self, values, aggregations):
        """`values` is a list of columns and `aggregations` is a list of list
        of aggregations. `aggregations[i]` is a list of aggregations for
        `values[i]`. Returns a tuple containing 1) list of list of aggregation
        results, 2) a list of grouped keys, and 3) a list of list of
        aggregations performed.
        """
        cdef vector[libcudf_groupby.scan_request] c_agg_requests
        cdef libcudf_groupby.scan_request c_agg_request
        cdef Column col
        cdef GroupbyScanAggregation agg_obj

        cdef pair[
            unique_ptr[table],
            vector[libcudf_groupby.aggregation_result]
        ] c_result

        allow_empty = all(len(v) == 0 for v in aggregations)

        included_aggregations = []
        column_included = set()
        for i, (col, aggs) in enumerate(zip(values, aggregations)):
            dtype = col.dtype

            valid_aggregations = (
                _LIST_AGGS if is_list_dtype(dtype)
                else _STRING_AGGS if is_string_dtype(dtype)
                else _CATEGORICAL_AGGS if is_categorical_dtype(dtype)
                else _STRUCT_AGGS if is_struct_dtype(dtype)
                else _INTERVAL_AGGS if is_interval_dtype(dtype)
                else _DECIMAL_AGGS if is_decimal_dtype(dtype)
                else "ALL"
            )
            included_aggregations_i = []

            c_agg_request = move(libcudf_groupby.scan_request())
            for agg in aggs:
                agg_obj = make_groupby_scan_aggregation(agg)
                if (valid_aggregations == "ALL"
                        or agg_obj.kind in valid_aggregations):
                    included_aggregations_i.append(agg)
                    c_agg_request.aggregations.push_back(
                        move(agg_obj.c_obj)
                    )
            included_aggregations.append(included_aggregations_i)
            if not c_agg_request.aggregations.empty():
                c_agg_request.values = col.view()
                c_agg_requests.push_back(
                    move(c_agg_request)
                )
                column_included.add(i)
        if c_agg_requests.empty() and not allow_empty:
            raise DataError("All requested aggregations are unsupported.")

        with nogil:
            c_result = move(
                self.c_obj.get()[0].scan(
                    c_agg_requests
                )
            )

        grouped_keys = columns_from_unique_ptr(
            move(c_result.first)
        )

        result_columns = _agg_result_from_columns(
            c_result.second, column_included, len(values)
        )

        return result_columns, grouped_keys, included_aggregations

    def aggregate(self, values, aggregations):
        """
        Parameters
        ----------
        values : Frame
        aggregations
            A dict mapping column names in `Frame` to a list of aggregations
            to perform on that column

            Each aggregation may be specified as:
            - a string (e.g., "max")
            - a lambda/function

        Returns
        -------
        Frame of aggregated values
        """
        if _is_all_scan_aggregate(aggregations):
            return self.scan_internal(values, aggregations)

        return self.aggregate_internal(values, aggregations)

    def shift(self, list values, int periods, list fill_values):
        cdef table_view view = table_view_from_columns(values)
        cdef size_type num_col = view.num_columns()
        cdef vector[size_type] offsets = vector[size_type](num_col, periods)

        cdef vector[reference_wrapper[constscalar]] c_fill_values
        cdef DeviceScalar d_slr
        d_slrs = []
        c_fill_values.reserve(num_col)
        for val, col in zip(fill_values, values):
            d_slr = as_device_scalar(val, dtype=col.dtype)
            d_slrs.append(d_slr)
            c_fill_values.push_back(
                reference_wrapper[constscalar](d_slr.get_raw_ptr()[0])
            )

        cdef pair[unique_ptr[table], unique_ptr[table]] c_result

        with nogil:
            c_result = move(
                self.c_obj.get()[0].shift(view, offsets, c_fill_values)
            )

        grouped_keys = columns_from_unique_ptr(move(c_result.first))
        shifted = columns_from_unique_ptr(move(c_result.second))

        return shifted, grouped_keys

    def replace_nulls(self, list values, object method):
        cdef table_view val_view = table_view_from_columns(values)
        cdef pair[unique_ptr[table], unique_ptr[table]] c_result
        cdef replace_policy policy = (
            replace_policy.PRECEDING
            if method == 'ffill' else replace_policy.FOLLOWING
        )
        cdef vector[replace_policy] policies = vector[replace_policy](
            val_view.num_columns(), policy
        )

        with nogil:
            c_result = move(
                self.c_obj.get()[0].replace_nulls(val_view, policies)
            )

        return columns_from_unique_ptr(move(c_result.second))


_GROUPBY_SCANS = {"cumcount", "cumsum", "cummin", "cummax", "rank"}


def _is_all_scan_aggregate(all_aggs):
    """
    Returns true if all are scan aggregations.
    Raises
    ------
    NotImplementedError
        If both reduction aggregations and scan aggregations are present.
    """

    def get_name(agg):
        return agg.__name__ if callable(agg) else agg

    all_scan = all(
        get_name(agg_name) in _GROUPBY_SCANS for aggs in all_aggs
        for agg_name in aggs
    )
    any_scan = any(
        get_name(agg_name) in _GROUPBY_SCANS for aggs in all_aggs
        for agg_name in aggs
    )

    if not all_scan and any_scan:
        raise NotImplementedError(
            "Cannot perform both aggregation and scan in one operation"
        )
    return all_scan and any_scan
