# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.scalar cimport Scalar


cpdef Column strip(Column input, side_type side, Scalar to_strip)
