# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from rmm.pylibrmm.stream cimport Stream


cpdef Column edit_distance(Column input, Column targets, Stream stream=*)

cpdef Column edit_distance_matrix(Column input, Stream stream=*)
