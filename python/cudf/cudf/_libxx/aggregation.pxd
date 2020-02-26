# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.string cimport string
from libcpp.memory cimport unique_ptr

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

from cudf._libxx.includes.aggregation cimport aggregation


cdef unique_ptr[aggregation] get_aggregation(op, kwargs) except *
