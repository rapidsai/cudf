# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

from libcpp.string cimport string

cdef extern from "binaryop.hpp" nogil:

    ctypedef enum gdf_binary_operator:
        GDF_ADD,
        GDF_SUB,
        GDF_MUL,
        GDF_DIV,
        GDF_TRUE_DIV,
        GDF_FLOOR_DIV,
        GDF_MOD,
        GDF_PYMOD,
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
        GDF_LOGICAL_AND,
        GDF_LOGICAL_OR,
        GDF_INVALID_BINARY

cdef extern from "binaryop.hpp" namespace "cudf" nogil:

    cdef void binary_operation(gdf_column* out, gdf_scalar* lhs, gdf_column* rhs, gdf_binary_operator ope) except +
    cdef void binary_operation(gdf_column* out, gdf_column* lhs, gdf_scalar* rhs, gdf_binary_operator ope) except +
    cdef void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, gdf_binary_operator ope) except +
    cdef void binary_operation(gdf_column* out, gdf_column* lhs, gdf_column* rhs, const string&       ptx) except +
