# Copyright (c) 2024-2025, NVIDIA CORPORATION.
from libcpp cimport bool
from pylibcudf.libcudf.labeling cimport inclusive

from .column cimport Column

from rmm.pylibrmm.stream cimport Stream


cpdef Column label_bins(
    Column input,
    Column left_edges,
    inclusive left_inclusive,
    Column right_edges,
    inclusive right_inclusive,
    Stream stream=*
)
