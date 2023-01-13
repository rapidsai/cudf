# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool

from cudf._lib.cpp cimport copying as cpp_copying

from .column_view cimport ColumnView
from .table cimport Table
from .table_view cimport TableView

ctypedef cbool underlying_type_t_out_of_bounds_policy
cpdef enum OutOfBoundsPolicy:
    NULLIFY = <underlying_type_t_out_of_bounds_policy> cpp_copying.NULLIFY
    DONT_CHECK = (
        <underlying_type_t_out_of_bounds_policy> cpp_copying.DONT_CHECK
    )


cdef cpp_copying.out_of_bounds_policy py_policy_to_c_policy(
    OutOfBoundsPolicy py_policy
) nogil


cpdef Table gather(
    TableView source_table,
    ColumnView gather_map,
    OutOfBoundsPolicy bounds_policy
)
