# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.cpp.groupby cimport (
    aggregation_request,
    aggregation_result,
    groupby,
)
from cudf._lib.cpp.table.table cimport table

from .aggregation cimport Aggregation
from .column cimport Column
from .table cimport Table


cdef class AggregationRequest:
    def __init__(self, Column values, list aggregations):
        self.values = values
        self.aggregations = aggregations

    cdef aggregation_request to_libcudf(self) except *:
        cdef aggregation_request c_obj
        c_obj.values = self.values.view()

        cdef Aggregation agg
        for agg in self.aggregations:
            c_obj.aggregations.push_back(move(agg.clone_underlying_as_groupby()))
        return move(c_obj)


cdef class GroupBy:
    def __init__(self, Table keys):
        self.c_obj.reset(new groupby(keys.view()))

    cpdef tuple aggregate(self, list requests):
        cdef AggregationRequest request
        cdef vector[aggregation_request] c_requests
        for request in requests:
            c_requests.push_back(move(request.to_libcudf()))

        cdef pair[unique_ptr[table], vector[aggregation_result]] c_res = move(
            dereference(self.c_obj).aggregate(c_requests)
        )
        cdef Table group_keys = Table.from_libcudf(move(c_res.first))

        cdef int i, j
        cdef list results = []
        cdef list inner_results
        for i in range(c_res.second.size()):
            inner_results = []
            for j in range(c_res.second[i].results.size()):
                inner_results.append(
                    Column.from_libcudf(move(c_res.second[i].results[j]))
                )
            results.append(inner_results)
        return group_keys, results
