# Copyright (c) 2023, NVIDIA CORPORATION.
import os

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.cpp.io.timezone cimport (
    build_timezone_transition_table as cpp_build_timezone_transition_table,
    make_optional,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.utils cimport columns_from_unique_ptr


def build_timezone_transition_table(tzdir, tzname):
    # TODO: libcudf needs the path to end with a '/' separator (but
    # shouldn't).  Remove this if/when that no longer a requirement:
    tzdir = os.path.join(tzdir, "")

    cdef unique_ptr[table] c_result
    cdef string c_tzdir = tzdir.encode()
    cdef string c_tzname = tzname.encode()

    with nogil:
        c_result = move(
            cpp_build_timezone_transition_table(
                make_optional[string](c_tzdir),
                c_tzname
            )
        )

    return columns_from_unique_ptr(move(c_result))
