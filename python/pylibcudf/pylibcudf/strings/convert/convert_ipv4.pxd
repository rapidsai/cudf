# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column cimport Column


cpdef Column ipv4_to_integers(Column input)

cpdef Column integers_to_ipv4(Column integers)

cpdef Column is_ipv4(Column input)
