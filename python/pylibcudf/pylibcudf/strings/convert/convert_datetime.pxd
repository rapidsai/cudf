# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.types cimport DataType


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
