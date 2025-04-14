# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table


cpdef Column interleave_columns(Table source_table)

cpdef Table tile(Table source_table, size_type count)

cpdef void table_to_array(
    Table table,
    DeviceBuffer output,
    DataType dtype,
    Stream stream=*
)
