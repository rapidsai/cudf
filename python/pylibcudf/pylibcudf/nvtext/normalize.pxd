# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.normalize cimport character_normalizer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

cdef class CharacterNormalizer:
    cdef unique_ptr[character_normalizer] c_obj

cpdef Column normalize_spaces(Column input, Stream stream=*, DeviceMemoryResource mr=*)

cpdef Column normalize_characters(
  Column input,
  CharacterNormalizer normalizer,
  Stream stream=*,
  DeviceMemoryResource mr=*
)
