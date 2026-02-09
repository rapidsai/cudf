# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.nvtext.tokenize cimport tokenize_vocabulary
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

cdef class TokenizeVocabulary:
    cdef unique_ptr[tokenize_vocabulary] c_obj

cpdef Column tokenize_scalar(
    Column input, Scalar delimiter=*, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column tokenize_column(
    Column input, Column delimiters, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column count_tokens_scalar(
    Column input, Scalar delimiter=*, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column count_tokens_column(
    Column input, Column delimiters, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column character_tokenize(
    Column input, Stream stream=*, DeviceMemoryResource mr=*
)

cpdef Column detokenize(
    Column input,
    Column row_indices,
    Scalar separator=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)

cpdef Column tokenize_with_vocabulary(
    Column input,
    TokenizeVocabulary vocabulary,
    Scalar delimiter,
    size_type default_id=*,
    Stream stream=*,
    DeviceMemoryResource mr=*,
)
