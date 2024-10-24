# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.datetime cimport (
    datetime_component,
    extract_datetime_component as cpp_extract_datetime_component,
    extract_year as cpp_extract_year,
)

from pylibcudf.libcudf.datetime import \
    datetime_component as DatetimeComponent  # no-cython-lint

from .column cimport Column


cpdef Column extract_year(
    Column values
):
    """
    Extract the year from a datetime column.

    Parameters
    ----------
    values : Column
        The column to extract the year from.

    Returns
    -------
    Column
        Column with the extracted years.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_year(values.view())
    return Column.from_libcudf(move(result))

cpdef Column extract_datetime_component(
    Column values,
    datetime_component component
):
    """
    Extract a datetime component from a datetime column.

    For details, see :cpp:func:`cudf::extract_datetime_component`.

    Parameters
    ----------
    values : Column
        The column to extract the component from.
    component : DatetimeComponent
        The datetime component to extract.

    Returns
    -------
    Column
        Column with the extracted component.
    """
    cdef unique_ptr[column] result

    with nogil:
        result = cpp_extract_datetime_component(values.view(), component)
    return Column.from_libcudf(move(result))
