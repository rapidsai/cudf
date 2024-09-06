# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.libcudf.datetime cimport datetime_component

from .column cimport Column


cpdef Column extract_year(
    Column col
)

cpdef Column extract_datetime_component(
    Column col,
    datetime_component component
)
