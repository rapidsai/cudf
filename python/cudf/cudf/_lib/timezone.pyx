# Copyright (c) 2023-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.optional cimport make_optional
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.io.timezone cimport (
    make_timezone_transition_table as cpp_make_timezone_transition_table,
)
from cudf._lib.pylibcudf.libcudf.table.table cimport table
from cudf._lib.utils cimport columns_from_unique_ptr


def make_timezone_transition_table(tzdir, tzname):
    cdef unique_ptr[table] c_result
    cdef string c_tzdir = tzdir.encode()
    cdef string c_tzname = tzname.encode()

    with nogil:
        c_result = move(
            cpp_make_timezone_transition_table(
                make_optional[string](c_tzdir),
                c_tzname
            )
        )

    return columns_from_unique_ptr(move(c_result))
