# Copyright (c) 2023, NVIDIA CORPORATION.

from pyarrow.lib cimport Table as pa_Table

from .table cimport Table


cpdef Table from_arrow(
    pa_Table pyarrow_table,
)
