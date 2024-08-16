# Copyright (c) 2024, NVIDIA CORPORATION.

from .column cimport Column


cpdef Column extract_year(
    Column col
)
