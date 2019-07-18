# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport cudf_table

cdef cudf_table* table_from_dataframe(df) except? NULL
cdef dataframe_from_table(cudf_table* table, colnames)
cdef columns_from_table(cudf_table* table)
cdef cudf_table* table_from_columns(columns) except? NULL
