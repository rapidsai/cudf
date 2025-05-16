# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.table.table cimport table


cdef extern from "cudf/timezone.hpp" namespace "cudf" nogil:
    unique_ptr[table] make_timezone_transition_table(
        optional[string] tzif_dir,
        string timezone_name
    ) except +libcudf_exception_handler
