# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.table.table_view cimport table_view


cdef extern from "cudf/lists/combine.hpp" namespace \
        "cudf::lists" nogil:

    ctypedef enum concatenate_null_policy:
        IGNORE "cudf::lists::concatenate_null_policy::IGNORE"
        NULLIFY_OUTPUT_ROW \
            "cudf::lists::concatenate_null_policy::NULLIFY_OUTPUT_ROW"

    cdef unique_ptr[column] concatenate_rows(
        const table_view input_table
    ) except +

    cdef unique_ptr[column] concatenate_list_elements(
        const table_view input_table,
    ) except +

    cdef unique_ptr[column] concatenate_list_elements(
        const column_view input_table,
        concatenate_null_policy null_policy
    ) except +
