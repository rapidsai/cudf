# Copyright (c) 2024, NVIDIA CORPORATION.

from .column cimport Column
from .table cimport Table


cpdef tuple inner_join(Table left_keys, Table right_keys)

cpdef tuple left_join(Table left_keys, Table right_keys)

cpdef tuple full_join(Table left_keys, Table right_keys)

cpdef Column left_semi_join(Table left_keys, Table right_keys)

cpdef Column left_anti_join(Table left_keys, Table right_keys)
