# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/reshape.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] interleave_columns(table_view source_table) \
        except +
    cdef unique_ptr[table] tile(table_view source_table, size_type count) \
        except +
