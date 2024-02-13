# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.lists cimport explode as cpp_explode
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type

from .table cimport Table


cpdef Table explode_outer(Table input, size_type explode_column_idx):
    cdef unique_ptr[table] c_result

    with nogil:
        c_result = move(cpp_explode.explode_outer(input.view(), explode_column_idx))

    return Table.from_libcudf(move(c_result))
