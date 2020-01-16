# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *
from cudf._lib.cudf cimport cudf_table
from libcpp.pair cimport pair
from libcpp.string cimport string


cdef table_to_dataframe(cudf_table* table, int_col_names=*)
cdef cudf_table* table_from_dataframe(df) except? NULL

cdef columns_from_table(cudf_table* table, int_col_names=*)
cdef cudf_table* table_from_columns(columns) except? NULL

cdef const unsigned char[::1] view_of_buffer(filepath_or_buffer) except *
