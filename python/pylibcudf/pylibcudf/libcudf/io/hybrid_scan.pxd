# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from libc.stddef cimport size_t
from libc.stdint cimport uint8_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view, mutable_column_view
from pylibcudf.libcudf.io.datasource cimport datasource
from pylibcudf.libcudf.io.parquet cimport parquet_reader_options
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from cuda.bindings.cyruntime cimport cudaStream_t
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.device_buffer cimport device_buffer
from rmm.librmm.memory_resource cimport device_async_resource_ref

ctypedef const uint8_t const_uint8_t
ctypedef const size_type const_size_type
ctypedef const device_span[const_uint8_t] const_device_span_const_uint8_t
ctypedef const byte_range_info const_byte_range_info

cdef extern from "cudf/io/experimental/hybrid_scan.hpp" \
        namespace "cudf::io::parquet::experimental" nogil:

    cpdef enum class use_data_page_mask(bool):
        YES
        NO

    cdef cppclass hybrid_scan_reader:
        hybrid_scan_reader(
            host_span[const_uint8_t] footer_bytes,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        hybrid_scan_reader(
            const FileMetaData& parquet_metadata,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        FileMetaData parquet_metadata() except +libcudf_exception_handler

        byte_range_info page_index_byte_range() except +libcudf_exception_handler

        void setup_page_index(
            host_span[const_uint8_t] page_index_bytes
        ) except +libcudf_exception_handler

        vector[size_type] all_row_groups(
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        size_type total_rows_in_row_groups(
            host_span[const_size_type] row_group_indices
        ) except +libcudf_exception_handler

        void reset_column_selection() except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_stats(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        pair[
            vector[byte_range_info], vector[byte_range_info]
        ] secondary_filters_byte_ranges(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_dictionary_pages(
            host_span[const_device_span_const_uint8_t] dictionary_page_data,
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        vector[size_type] filter_row_groups_with_bloom_filters(
            host_span[const_device_span_const_uint8_t] bloom_filter_data,
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream
        ) except +libcudf_exception_handler

        unique_ptr[column] build_row_mask_with_page_index_stats(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] filter_column_chunks_byte_ranges(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_filter_columns(
            host_span[const_size_type] row_group_indices,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            mutable_column_view& row_mask,
            use_data_page_mask mask_data_pages,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] payload_column_chunks_byte_ranges(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_payload_columns(
            host_span[const_size_type] row_group_indices,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        vector[byte_range_info] all_column_chunks_byte_ranges(
            host_span[const_size_type] row_group_indices,
            const parquet_reader_options& options
        ) except +libcudf_exception_handler

        table_with_metadata materialize_all_columns(
            host_span[const_size_type] row_group_indices,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        void setup_chunking_for_filter_columns(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            host_span[const_size_type] row_group_indices,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        table_with_metadata materialize_filter_columns_chunk(
            mutable_column_view& row_mask
        ) except +libcudf_exception_handler

        void setup_chunking_for_payload_columns(
            size_t chunk_read_limit,
            size_t pass_read_limit,
            host_span[const_size_type] row_group_indices,
            const column_view& row_mask,
            use_data_page_mask mask_data_pages,
            host_span[const_device_span_const_uint8_t] column_chunk_data,
            const parquet_reader_options& options,
            cudaStream_t stream,
            device_async_resource_ref mr
        ) except +libcudf_exception_handler

        table_with_metadata materialize_payload_columns_chunk(
            const column_view& row_mask
        ) except +libcudf_exception_handler

        vector[vector[size_type]] construct_row_group_passes(
            host_span[const_size_type] row_group_indices,
            size_t pass_read_limit,
        ) except +libcudf_exception_handler

        bool has_next_table_chunk() except +libcudf_exception_handler


# Bloom filter fetch IO util (cudf/io/parquet_io_utils.hpp). Co-located here as
# the hybrid scan flow is its only consumer. Binding the tuple/future return
# requires a little std:: glue since there is no Cython precedent for either.
cdef extern from "cudf/utilities/span.hpp" namespace "cudf" nogil:
    cdef cppclass device_span_u8 "cudf::device_span<const uint8_t>":
        const_uint8_t* data()
        size_t size()


cdef extern from "<future>" namespace "std" nogil:
    cdef cppclass future_void "std::future<void>":
        void get() except +libcudf_exception_handler


cdef extern from "cudf/io/parquet_io_utils.hpp" namespace "cudf::io::parquet" nogil:
    cdef cppclass bloom_filter_fetch_result "std::tuple<std::vector<rmm::device_buffer>, std::vector<cudf::device_span<const uint8_t> >, std::future<void> >":  # noqa: E501
        bloom_filter_fetch_result()

    bloom_filter_fetch_result fetch_bloom_filters_to_device_async(
        datasource& source,
        host_span[const_byte_range_info] bloom_filter_byte_ranges,
        cuda_stream_view stream,
        device_async_resource_ref mr,
    ) except +libcudf_exception_handler


cdef extern from "<tuple>" namespace "std" nogil:
    vector[device_buffer]& bloom_fetch_get_buffers "std::get<0>"(
        bloom_filter_fetch_result& result
    )
    vector[device_span_u8]& bloom_fetch_get_spans "std::get<1>"(
        bloom_filter_fetch_result& result
    )
    future_void& bloom_fetch_get_future "std::get<2>"(
        bloom_filter_fetch_result& result
    )
