# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column


cpdef Column url_encode(Column Input)

cpdef Column url_decode(Column Input)
