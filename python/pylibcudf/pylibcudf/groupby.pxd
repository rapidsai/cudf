# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.libcudf.aggregation cimport (
    aggregation,
    groupby_aggregation,
    groupby_scan_aggregation,
)
from pylibcudf.libcudf.groupby cimport (
    aggregation_request,
    aggregation_result,
    groupby,
    scan_request,
)
from pylibcudf.libcudf.table.table cimport table
from pylibcudf.libcudf.types cimport null_order, order

from rmm.pylibrmm.stream cimport Stream

from .column cimport Column
from .table cimport Table


cdef class GroupByRequest:
    # The groupby APIs accept vectors of unique_ptrs to aggregation requests.
    # This ownership model means that if GroupByRequest owned the
    # corresponding C++ object, that object would have to be copied by e.g.
    # each groupby.aggregate call to avoid invalidating this object. Therefore,
    # this class instead stores only Python/Cython objects and constructs the
    # C++ object on the fly as requested.
    cdef Column _values
    cdef list _aggregations

    cdef aggregation_request _to_libcudf_agg_request(self) except *
    cdef scan_request _to_libcudf_scan_request(self) except *


cdef class GroupBy:
    cdef unique_ptr[groupby] c_obj
    cdef Table _keys
    cdef unique_ptr[vector[order]] _column_order
    cdef unique_ptr[vector[null_order]] _null_precedence

    cpdef tuple aggregate(self, list requests, Stream stream=*)
    cpdef tuple scan(self, list requests, Stream stream=*)
    cpdef tuple shift(
        self,
        Table values,
        list offset,
        list fill_values,
        Stream stream=*
    )
    cpdef tuple replace_nulls(
        self,
        Table values,
        list replace_policies,
        Stream stream=*
    )
    cpdef tuple get_groups(self, Table values=*, Stream stream=*)

    @staticmethod
    cdef tuple _parse_outputs(
        pair[unique_ptr[table], vector[aggregation_result]] c_res,
        Stream stream,
    )
