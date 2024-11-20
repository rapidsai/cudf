# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.optional cimport make_optional
from libcpp.string cimport string
from libcpp.utility cimport move
from pylibcudf.libcudf.io.timezone cimport (
    make_timezone_transition_table as cpp_make_timezone_transition_table,
)
from pylibcudf.libcudf.table.table cimport table

from ..table cimport Table

__all__ = ["make_timezone_transition_table"]

cpdef Table make_timezone_transition_table(str tzif_dir, str timezone_name):
    """
    Creates a transition table to convert ORC timestamps to UTC.

    Parameters
    ----------
    tzif_dir : str
        The directory where the TZif files are located
    timezone_name : str
        standard timezone name

    Returns
    -------
    Table
        The transition table for the given timezone.
    """
    cdef unique_ptr[table] c_result
    cdef string c_tzdir = tzif_dir.encode()
    cdef string c_tzname = timezone_name.encode()

    with nogil:
        c_result = cpp_make_timezone_transition_table(
            make_optional[string](c_tzdir),
            c_tzname
        )

    return Table.from_libcudf(move(c_result))
