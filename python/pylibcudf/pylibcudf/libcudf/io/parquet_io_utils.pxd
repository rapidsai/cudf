# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_async_resource_ref

from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.utilities.span cimport device_span, host_span

ctypedef const uint8_t const_uint8_t
ctypedef const byte_range_info const_byte_range_info

cdef extern from "cudf/io/parquet_io_utils.hpp" \
        namespace "cudf::io::parquet" nogil:

    pair[vector[device_buffer], vector[device_span[const_uint8_t]]] \
        fetch_byte_ranges_to_device(
            datasource& source,
            host_span[const_byte_range_info] byte_ranges,
            cuda_stream_view stream,
            device_async_resource_ref mr,
        ) except +libcudf_exception_handler

    unique_ptr[datasource.buffer] fetch_page_index_to_host(
        datasource& ds,
        byte_range_info page_index_bytes,
    ) except +libcudf_exception_handler
