# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._lib.cpp.io.types cimport (
    source_info,
    sink_info,
    data_sink,
    column_name_info
)
from cudf._lib.table cimport Table

cdef source_info make_source_info(list src) except*
cdef sink_info make_sink_info(src, unique_ptr[data_sink] & data) except*
cdef update_struct_field_names(
    Table table,
    vector[column_name_info]& schema_info)
