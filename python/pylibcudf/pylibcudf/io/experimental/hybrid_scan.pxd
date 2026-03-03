# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.io.hybrid_scan cimport (
    hybrid_scan_reader as cpp_hybrid_scan_reader,
    use_data_page_mask,
)
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.libcudf.io.hybrid_scan cimport const_uint8_t
from pylibcudf.libcudf.utilities.span cimport device_span


cdef device_span[const_uint8_t] _get_device_span(object obj) except *


cdef class FileMetaData:
    cdef cpp_FileMetaData c_obj

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata)


cdef class HybridScanReader:
    cdef unique_ptr[cpp_hybrid_scan_reader] c_obj
    cdef Stream stream
    cdef DeviceMemoryResource mr
