# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string

from cudf._lib.pylibcudf.column cimport Column
from cudf._lib.pylibcudf.types cimport DataType


cpdef Column to_durations(
    Column input,
    DataType duration_type,
    const string& format
)

cpdef Column from_durations(
    Column input,
    const string& format
)
