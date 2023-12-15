# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool

from cudf._lib.cpp.copying cimport out_of_bounds_policy

from .column cimport Column
from .table cimport Table


cpdef Table gather(
    Table source_table,
    Column gather_map,
    out_of_bounds_policy bounds_policy
)
