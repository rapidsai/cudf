# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp cimport bool
from pylibcudf.column cimport Column


cpdef Column normalize_spaces(Column input)

cpdef Column normalize_characters(Column input, bool do_lower_case)
