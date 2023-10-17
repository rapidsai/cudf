# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.libcpp.optional cimport optional
from cudf._lib.cpp.table.table cimport table


cdef extern from "cudf/timezone.hpp" namespace "cudf" nogil:
    unique_ptr[table] make_timezone_transition_table(
        optional[string] tzif_dir,
        string timezone_name
    ) except +
