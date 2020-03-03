# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/sorting.hpp" namespace "cudf::experimental" nogil:
    cdef bool is_sorted(
        table_view table,
        vector[order] column_order,
        vector[null_order] null_precedence) except +

