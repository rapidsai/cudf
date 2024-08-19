# Copyright (c) 2024, NVIDIA CORPORATION.

from .table cimport Table


cpdef Table merge (
    list tables_to_merge,
    list key_cols,
    list column_order,
    list null_precedence,
)
