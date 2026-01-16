# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# ---------------------- Imports ----------------------
from libcudf.strings.split cimport split_part as cpp_split_part
from libcudf.strings.strings_column_view cimport strings_column_view
from libcudf.scalar.scalar cimport string_scalar
from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.types cimport size_type
from libcpp.memory cimport unique_ptr
from libcudf cimport move

# ---------------------- The Function ----------------------
cpdef Column split_part(Column strings, Scalar delimiter, size_type index):
    cdef strings_column_view c_strings = strings.view()
    cdef const string_scalar* c_delim = <const string_scalar*>delimiter.c_value()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = cpp_split_part(
            c_strings,
            c_delim[0],
            index,
            <rmm.mr.device_memory_resource*>NULL  # default MR
        )

    return Column.from_libcudf(move(c_result))