# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *
from enum import Enum


class InterpolationEnum(Enum):
    LINEAR   = <underlying_type_t_interpolation> interpolation.LINEAR
    LOWER    = <underlying_type_t_interpolation> interpolation.LOWER
    HIGHER   = <underlying_type_t_interpolation> interpolation.HIGHER
    MIDPOINT = <underlying_type_t_interpolation> interpolation.MIDPOINT
    NEAREST  = <underlying_type_t_interpolation> interpolation.NEAREST

class OrderEnum(Enum):
    ASCENDING  = <underlying_type_t_order> order.ASCENDING
    DESCENDING = <underlying_type_t_order> order.DESCENDING

class SortedEnum(Enum):
    YES = <underlying_type_t_sorted> sorted.YES
    NO  = <underlying_type_t_sorted> sorted.NO

class NullOrderEnum(Enum):
    BEFORE = <underlying_type_t_order> null_order.BEFORE
    AFTER  = <underlying_type_t_order> null_order.AFTER
