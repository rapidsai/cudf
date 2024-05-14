# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.libcudf.io.data_sink cimport data_sink
from cudf._lib.pylibcudf.libcudf.io.types cimport (
    column_name_info,
    sink_info,
    source_info,
)


cdef source_info make_source_info(list src) except*
cdef sink_info make_sinks_info(
    list src, vector[unique_ptr[data_sink]] & data) except*
cdef sink_info make_sink_info(src, unique_ptr[data_sink] & data) except*
cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info)
cdef Column update_column_struct_field_names(
    Column col,
    column_name_info& info
)
