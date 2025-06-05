# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector
from pylibcudf.libcudf.scalar.scalar cimport scalar
from rmm.pylibrmm.stream cimport Stream

cdef vector[reference_wrapper[const scalar]] _as_vector(list source)
cdef Stream _get_stream(Stream stream = *)
