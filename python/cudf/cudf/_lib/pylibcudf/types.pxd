# Copyright (c) 2022, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport type_id


cdef type_id py_type_to_c_type(py_type_id)
