# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.functional cimport reference_wrapper
from libcpp.vector cimport vector
from pylibcudf.libcudf.scalar.scalar cimport scalar
from rmm.pylibrmm.stream cimport Stream
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource

cdef vector[reference_wrapper[const scalar]] _as_vector(list source)
cpdef Stream _get_stream(Stream stream = *)
cdef DeviceMemoryResource _get_memory_resource(DeviceMemoryResource mr = *)
