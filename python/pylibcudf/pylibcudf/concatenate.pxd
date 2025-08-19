# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from .table cimport Table

from rmm.pylibrmm.stream cimport Stream


# There is no way to define a fused type that is a list of other objects, so we cannot
# unify the column and table paths without using runtime dispatch instead. In this case
# we choose to prioritize API consistency over performance, so we use the same function
# with a bit of runtime dispatch overhead.
cpdef concatenate(list objects, Stream stream=*)
