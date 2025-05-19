# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.libcudf.json cimport get_json_object_options
from pylibcudf.scalar cimport Scalar


cdef class GetJsonObjectOptions:
    cdef get_json_object_options options


cpdef Column get_json_object(
    Column col,
    Scalar json_path,
    GetJsonObjectOptions options=*
)
