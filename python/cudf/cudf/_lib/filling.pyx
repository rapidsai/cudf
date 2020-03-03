# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *
from cudf._lib.cudf import *

from cudf._lib.utils cimport (
    columns_from_table,
    table_from_columns
)
from cudf._lib.includes.filling cimport (
    tile as cpp_tile
)

import numpy as np


def tile(input, count):
    from cudf.core.column import as_column

    cdef cudf_table* c_input_table = table_from_columns(input)
    cdef cudf_table c_result_table

    cdef size_type c_count = count

    if np.isscalar(count):
        with nogil:
            c_result_table = cpp_tile(
                c_input_table[0],
                c_count)
    else:
        raise ValueError("Count has to be numerical scalar")

    del c_input_table

    return columns_from_table(&c_result_table)
