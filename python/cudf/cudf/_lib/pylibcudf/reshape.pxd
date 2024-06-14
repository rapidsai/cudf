# Copyright (c) 2024, NVIDIA CORPORATION.

from cudf._lib.pylibcudf.libcudf.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table


cpdef Column interleave_columns(Table source_table)
cpdef Table tile(Table source_table, size_type count)
