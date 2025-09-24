# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.json cimport get_json_object_options
from pylibcudf.scalar cimport Scalar

from rmm.pylibrmm.stream cimport Stream


cdef class GetJsonObjectOptions:
    cdef get_json_object_options options


cpdef Column get_json_object(
    Column col,
    Scalar json_path,
    GetJsonObjectOptions options=*,
    Stream stream=*
)
