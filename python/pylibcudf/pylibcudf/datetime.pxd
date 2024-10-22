# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.datetime cimport datetime_component

from .column cimport Column


cpdef Column extract_year(
    Column col
)

cpdef Column extract_month(
    Column col
)

cpdef Column extract_day(
    Column col
)

cpdef Column extract_weekday(
    Column col
)

cpdef Column extract_hour(
    Column col
)

cpdef Column extract_minute(
    Column col
)

cpdef Column extract_second(
    Column col
)

cpdef Column extract_millisecond_fraction(
    Column col
)

cpdef Column extract_microsecond_fraction(
    Column col
)

cpdef Column extract_nanosecond_fraction(
    Column col
)

cpdef Column extract_datetime_component(
    Column col,
    datetime_component component
)
