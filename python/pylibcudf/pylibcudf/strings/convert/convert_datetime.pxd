# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.types cimport DataType


cpdef Column to_timestamps(
    Column input,
    DataType timestamp_type,
    const string& format
)

cpdef Column from_timestamps(
    Column input,
    const string& format,
    Column input_strings_names
)
