# Copyright (c) 2024, NVIDIA CORPORATION.

from .column cimport Column
from .gpumemoryview cimport gpumemoryview


cpdef tuple[gpumemoryview, int] nans_to_nulls(Column input)
