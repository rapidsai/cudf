# Copyright (c) 2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream cimport Stream

cpdef void join_streams(list streams, Stream stream)
