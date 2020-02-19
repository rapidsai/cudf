# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from enum import Enum


class InterpolationEnum(Enum):
    linear   = <int32_t> interpolation.LINEAR
    lower    = <int32_t> interpolation.LOWER
    higher   = <int32_t> interpolation.HIGHER
    midpoint = <int32_t> interpolation.MIDPOINT
    nearest  = <int32_t> interpolation.NEAREST

class OrderEnum(Enum):
    ascending  = <bool> order.ASCENDING
    descending = <bool> order.DESCENDING

class SortedEnum(Enum):
    yes = <bool> sorted.YES
    no  = <bool> sorted.NO

class NullOrderEnum(Enum):
    before = <bool> null_order.BEFORE
    after  = <bool> null_order.AFTER
