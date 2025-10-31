# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
from rmm.librmm.device_buffer cimport device_buffer


cdef class FileMetaData:
    cdef cpp_FileMetaData c_obj

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata)


cdef class HybridScanReader:
    cdef unique_ptr[cpp_hybrid_scan_reader] c_obj


cdef class DeviceBuffer:
    cdef device_buffer c_obj
