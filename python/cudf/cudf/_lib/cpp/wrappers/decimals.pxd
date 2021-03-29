from libc.stdint cimport int64_t, int32_t

cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    ctypedef enum Radix:
        BASE_10 "numeric::Radix::BASE_10"

    cdef cppclass scale_type:
        scale_type(int32_t)
        int32_t operator()() except +

    cdef cppclass fixed_point[T, U]:
        int32_t value() except +
        int64_t value() except +
        scale_type scale() except +
    
    ctypedef fixed_point[int64_t, Radix.BASE_10] decimal64
