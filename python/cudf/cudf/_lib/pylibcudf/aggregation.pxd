# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp cimport aggregation as cpp_aggregation
from cudf._lib.cpp.aggregation cimport Kind as kind_t, aggregation
from cudf._lib.cpp.types cimport null_policy


cdef class Aggregation:
    cdef aggregation *c_ptr
    cpdef kind_t kind(self)


ctypedef unique_ptr[aggregation](*nullary_factory_type)() except +
ctypedef unique_ptr[aggregation](*unary_null_handling_factory_type)(
    null_policy
) except +


cdef Aggregation _create_nullary_agg(cls, nullary_factory_type cpp_agg_factory)
cdef Aggregation _create_unary_null_handling_agg(
    cls,
    unary_null_handling_factory_type cpp_agg_factory,
    null_policy null_handling,
)
