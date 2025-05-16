# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column
from pylibcudf.scalar cimport Scalar


cpdef Column to_booleans(Column input, Scalar true_string)

cpdef Column from_booleans(Column booleans, Scalar true_string, Scalar false_string)
