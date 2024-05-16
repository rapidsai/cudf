# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport string_scalar
from cudf._lib.pylibcudf.libcudf.strings.convert.convert_lists cimport (
    format_list_column as cpp_format_list_column,
)

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar


@acquire_spill_lock()
def format_list_column(Column source_list, Column separators):
    """
    Format a list column of strings into a strings column.

    Parameters
    ----------
    input_col : input column of type list with strings child.

    separators: strings used for formatting (', ', '[', ']')

    Returns
    -------
    Formatted strings column
    """
    cdef unique_ptr[column] c_result
    cdef column_view source_view = source_list.view()
    cdef column_view separators_view = separators.view()
    # Use 'None' as null-replacement string
    cdef DeviceScalar str_na_rep = as_device_scalar("None")
    cdef const string_scalar* string_scalar_na_rep = <const string_scalar*>(
        str_na_rep.get_raw_ptr())

    with nogil:
        c_result = move(cpp_format_list_column(
            source_view, string_scalar_na_rep[0], separators_view
        ))

    return Column.from_unique_ptr(
        move(c_result)
    )
