# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from .table cimport Table

from rmm.pylibrmm.stream cimport Stream


cpdef Table merge (
    list tables_to_merge,
    list key_cols,
    list column_order,
    list null_precedence,
    Stream stream=*
)
