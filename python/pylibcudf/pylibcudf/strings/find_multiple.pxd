# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.table cimport Table
from rmm.pylibrmm.stream cimport Stream


cpdef Column find_multiple(Column input, Column targets, Stream stream=*)
cpdef Table contains_multiple(Column input, Column targets, Stream stream=*)
