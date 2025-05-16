# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column


cpdef Column edit_distance(Column input, Column targets)

cpdef Column edit_distance_matrix(Column input)
