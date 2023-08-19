# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp cimport bool as cbool

from cudf._lib.cpp cimport copying as cpp_copying

from .column cimport Column
from .table cimport Table


cpdef Table gather(
    Table source_table,
    Column gather_map,
    cpp_copying.out_of_bounds_policy bounds_policy
)
