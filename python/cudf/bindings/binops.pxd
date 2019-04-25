# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    ctypedef enum gdf_binary_operator:
        GDF_ADD,
        GDF_SUB,
        GDF_MUL,
        GDF_DIV,
        GDF_TRUE_DIV,
        GDF_FLOOR_DIV,
        GDF_MOD,
        GDF_POW,
        GDF_EQUAL,
        GDF_NOT_EQUAL,
        GDF_LESS,
        GDF_GREATER,
        GDF_LESS_EQUAL,
        GDF_GREATER_EQUAL,
        GDF_BITWISE_AND,
        GDF_BITWISE_OR,
        GDF_BITWISE_XOR,

    cdef gdf_error gdf_binary_operation_s_v(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) except +
    cdef gdf_error gdf_binary_operation_v_s(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) except +
    cdef gdf_error gdf_binary_operation_v_v(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) except +

    cdef gdf_error gdf_add_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_sub_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_mul_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_floordiv_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_div_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_gt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_ge_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_lt_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_le_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_eq_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_ne_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_bitwise_and_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_bitwise_or_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
    cdef gdf_error gdf_bitwise_xor_generic(gdf_column *lhs, gdf_column *rhs, gdf_column *output) except +
