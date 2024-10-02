# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column


cpdef Column find_multiple(Column input, Column targets)
