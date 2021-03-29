from libc.stdint cimport int64_t, int32_t

cdef extern from "cudf/fixed_point/fixed_point.hpp" namespace "numeric" nogil:
    ctypedef int64_t decimal64

    cdef cppclass scale_type:
        scale_type(int32_t)
        int32_t operator()() except +
