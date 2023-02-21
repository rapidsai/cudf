# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.cpp.io.timezone cimport (
    build_timezone_transition_table as cpp_build_timezone_transition_table,
    nullopt,
    optional,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport columns_from_unique_ptr


def build_timezone_transition_table(timezone_name, tzif_dir):

    cdef unique_ptr[table] c_result
    cdef optional[string] c_tzif_dir = nullopt
    cdef string c_timezone_name = timezone_name.encode()

    with nogil:
        c_result = move(
            cpp_build_timezone_transition_table(
                c_tzif_dir,
                c_timezone_name
            )
        )

    return columns_from_unique_ptr(move(c_result))
