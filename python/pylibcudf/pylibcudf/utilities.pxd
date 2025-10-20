# Copyright (c) 2025, NVIDIA CORPORATION.

from libcpp cimport bool
from rmm.pylibrmm.stream cimport Stream

cpdef bool is_ptds_enabled()
cpdef void join_streams(list streams, Stream stream)
