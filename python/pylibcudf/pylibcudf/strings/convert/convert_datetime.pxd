# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    str format
)

cpdef Column from_timestamps(
    Column timestamps,
    str format,
    Column input_strings_names
)

cpdef Column is_timestamp(
    Column input,
    str format,
)
