# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings.char_types cimport string_character_types
from pylibcudf.scalar cimport Scalar
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream


cpdef Column all_characters_of_type(
    Column source_strings,
    string_character_types types,
    string_character_types verify_types,
    Stream stream=*,
    DeviceMemoryResource mr=*
)

cpdef Column filter_characters_of_type(
    Column source_strings,
    string_character_types types_to_remove,
    Scalar replacement,
    string_character_types types_to_keep,
    Stream stream=*,
    DeviceMemoryResource mr=*
)
