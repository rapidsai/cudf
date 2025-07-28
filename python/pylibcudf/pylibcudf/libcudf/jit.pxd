from libc.stdint cimport int8_t



cdef extern from "cudf/jit/udf.hpp" namespace "cudf" nogil:

    cpdef enum class udf_source_type(int8_t):
        CUDA
        PTX