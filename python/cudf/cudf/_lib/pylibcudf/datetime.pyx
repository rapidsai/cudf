# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.datetime cimport (
    day_of_year as cpp_day_of_year,
    extract_day as cpp_extract_day,
    extract_hour as cpp_extract_hour,
    extract_microsecond_fraction as cpp_extract_microsecond_fraction,
    extract_millisecond_fraction as cpp_extract_millisecond_fraction,
    extract_minute as cpp_extract_minute,
    extract_month as cpp_extract_month,
    extract_nanosecond_fraction as cpp_extract_nanosecond_fraction,
    extract_second as cpp_extract_second,
    extract_weekday as cpp_extract_weekday,
    extract_year as cpp_extract_year,
)

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
        result = move(cpp_extract_year(values.view()))
    return Column.from_libcudf(move(result))


def extract_datetime_component(Column col, str field):

    cdef unique_ptr[column] c_result

    with nogil:
        if field == "year":
            c_result = move(cpp_extract_year(col.view()))
        elif field == "month":
            c_result = move(cpp_extract_month(col.view()))
        elif field == "day":
            c_result = move(cpp_extract_day(col.view()))
        elif field == "weekday":
            c_result = move(cpp_extract_weekday(col.view()))
        elif field == "hour":
            c_result = move(cpp_extract_hour(col.view()))
        elif field == "minute":
            c_result = move(cpp_extract_minute(col.view()))
        elif field == "second":
            c_result = move(cpp_extract_second(col.view()))
        elif field == "millisecond":
            c_result = move(
                cpp_extract_millisecond_fraction(col.view())
            )
        elif field == "microsecond":
            c_result = move(
                cpp_extract_microsecond_fraction(col.view())
            )
        elif field == "nanosecond":
            c_result = move(
                cpp_extract_nanosecond_fraction(col.view())
            )
        elif field == "day_of_year":
            c_result = move(cpp_day_of_year(col.view()))
        else:
            raise ValueError(f"Invalid datetime field: '{field}'")

    return Column.from_libcudf(move(c_result))
