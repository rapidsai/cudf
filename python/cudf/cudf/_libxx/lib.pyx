# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from enum import Enum


class InterpolationEnum(Enum):
    LINEAR   = <int32_t> interpolation.LINEAR
    LOWER    = <int32_t> interpolation.LOWER
    HIGHER   = <int32_t> interpolation.HIGHER
    MIDPOINT = <int32_t> interpolation.MIDPOINT
    NEAREST  = <int32_t> interpolation.NEAREST

class OrderEnum(Enum):
    ASCENDING  = <bool> order.ASCENDING
    DESCENDING = <bool> order.DESCENDING

class SortedEnum(Enum):
    YES = <bool> sorted.YES
    NO  = <bool> sorted.NO

class NullOrderEnum(Enum):
    BEFORE = <bool> null_order.BEFORE
    AFTER  = <bool> null_order.AFTER
