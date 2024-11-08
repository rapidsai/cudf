# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_durations(
    Column input,
    DataType duration_type,
    str format
)

cpdef Column from_durations(
    Column durations,
    str format=*
)
