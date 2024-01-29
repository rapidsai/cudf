# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp cimport aggregation as cpp_aggregation
from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    groupby_aggregation,
)
from cudf._lib.cpp.types cimport null_policy

ctypedef groupby_aggregation * gba_ptr

cdef class Aggregation:
    cdef aggregation *c_ptr
    cpdef kind_t kind(self)
    cdef unique_ptr[groupby_aggregation] make_groupby_copy(self) except *


ctypedef unique_ptr[aggregation](*nullary_factory_type)() except +
ctypedef unique_ptr[aggregation](*unary_null_handling_factory_type)(
    null_policy
) except +


cdef Aggregation _create_nullary_agg(nullary_factory_type cpp_agg_factory)
cdef Aggregation _create_unary_null_handling_agg(
    unary_null_handling_factory_type cpp_agg_factory,
    null_policy null_handling,
)
