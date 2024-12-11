# Copyright (c) 2024, NVIDIA CORPORATION.

from libcpp.string cimport string
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.side_type cimport side_type
from pylibcudf.libcudf.types cimport size_type


cpdef Column pad(Column input, size_type width, side_type side, str fill_char)

cpdef Column zfill(Column input, size_type width)
