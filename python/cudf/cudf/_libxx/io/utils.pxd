# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.io.types cimport source_info

cdef source_info make_source_info(filepath_or_buffer) except*
