# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool

from .column cimport Column


cpdef Column label_bins(
    Column input,
    Column left_edges,
    bool left_inclusive,
    Column right_edges,
    bool right_inclusive
)
