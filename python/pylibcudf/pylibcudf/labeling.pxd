# Copyright (c) 2024, NVIDIA CORPORATION.
from libcpp cimport bool
from pylibcudf.libcudf.labeling cimport inclusive

from .column cimport Column


cpdef Column label_bins(
    Column input,
    Column left_edges,
    inclusive left_inclusive,
    Column right_edges,
    inclusive right_inclusive
)
