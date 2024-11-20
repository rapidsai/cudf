# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from pylibcudf.libcudf.io.data_sink cimport data_sink
from pylibcudf.libcudf.io.types cimport (
    column_name_info,
    sink_info,
    source_info,
)

from cudf._lib.column cimport Column


cdef sink_info make_sinks_info(
    list src, vector[unique_ptr[data_sink]] & data) except*
cdef sink_info make_sink_info(src, unique_ptr[data_sink] & data) except*
cdef add_df_col_struct_names(
    df,
    child_names_dict
)
cdef update_col_struct_field_names(
    Column col,
    child_names
)
cdef update_struct_field_names(
    table,
    vector[column_name_info]& schema_info)
cdef Column update_column_struct_field_names(
    Column col,
    column_name_info& info
)
