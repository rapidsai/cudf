# Copyright (c) 2023, NVIDIA CORPORATION.

from cudf._lib.cpp.copying cimport out_of_bounds_policy

from .column_view cimport ColumnView
from .table cimport Table
from .table_view cimport TableView


cdef Table gather(
    TableView source_table,
    ColumnView gather_map,
    out_of_bounds_policy bounds_policy
)
