# Copyright (c) 2024, NVIDIA CORPORATION.

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from pylibcudf.column cimport Column
from pylibcudf.libcudf.strings_udf cimport udf_string
from pylibcudf.libcudf.types cimport size_type

from rmm.pylibrmm.device_buffer cimport DeviceBuffer


cpdef int get_cuda_build_version()

cpdef DeviceBuffer column_to_string_view_array(Column input)

cdef Column column_from_udf_string_array(udf_string* input, size_type size)

cdef void free_udf_string_array(udf_string* input, size_type size)

cpdef uintptr_t get_character_flags_table()

cpdef uintptr_t get_character_cases_table()

cpdef uintptr_t get_special_case_mapping_table()
