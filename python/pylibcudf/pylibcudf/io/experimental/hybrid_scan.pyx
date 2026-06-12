# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cython.operator cimport dereference
from libc.stdint cimport uint8_t, uintptr_t
from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.io.parquet_metadata cimport FileMetaData as c_FileMetaData
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.io.text cimport ByteRangeInfo
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view, mutable_column_view
from pylibcudf.libcudf.io.datasource cimport datasource, make_datasources
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_device_span_const_uint8_t,
    const_size_type,
    const_uint8_t,
    hybrid_scan_read_options as cpp_hybrid_scan_read_options,
    hybrid_scan_reader as cpp_hybrid_scan_reader,
    use_data_page_mask as cpp_use_data_page_mask,
)
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.io.types cimport SourceInfo
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport device_span, host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream

from pylibcudf.span import is_span
from pylibcudf.io.parquet_metadata import FileMetaData

import pylibcudf.libcudf.io.hybrid_scan

UseDataPageMask = pylibcudf.libcudf.io.hybrid_scan.use_data_page_mask

__all__ = ["FileMetaData", "HybridScanReader", "UseDataPageMask"]


cdef device_span[const_uint8_t] _get_device_span(object obj) except *:
    """Convert a Span-like object to a device_span<const uint8_t>."""
    if not is_span(obj):
        raise TypeError(
            f"Object of type {type(obj)} does not implement the Span protocol"
        )
    return device_span[const_uint8_t](<const_uint8_t*>
                                      <uintptr_t>obj.ptr,
                                      <size_t>obj.size)


cdef class HybridScanReader:
    """Experimental Parquet reader optimized for highly selective filters.

    This class implements a hybrid scan operation for reading Parquet files
    with highly selective filters. It reads in two passes: first reading
    filter columns to build a row mask, then reading payload columns using
    that mask for optimization.

    For details, see :cpp:class:`cudf::io::parquet::experimental::hybrid_scan_reader`

    Parameters
    ----------
    footer_bytes : Buffer
        Parquet file footer bytes
    options : ParquetReaderOptions
        Parquet reader options

    Examples
    --------
    >>> import pylibcudf as plc
    >>> # Create reader from footer bytes
    >>> reader = plc.io.hybrid_scan.HybridScanReader(footer_bytes, options)
    >>> # Get metadata
    >>> metadata = reader.parquet_metadata()
    >>> # Get all row groups
    >>> row_groups = reader.all_row_groups(options)
    """

    def __init__(self, const uint8_t[::1] footer_bytes, ParquetReaderOptions options):
        with nogil:
            self.c_obj = make_unique[cpp_hybrid_scan_reader](
                host_span[const_uint8_t](&footer_bytes[0], len(footer_bytes)),
                options.c_obj
            )

    @staticmethod
    def from_parquet_metadata(c_FileMetaData metadata, ParquetReaderOptions options):
        """Create a HybridScanReader from pre-populated metadata.

        Parameters
        ----------
        metadata : FileMetaData
            Pre-populated Parquet file metadata
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        HybridScanReader
        """
        cdef HybridScanReader reader = HybridScanReader.__new__(HybridScanReader)
        with nogil:
            reader.c_obj = make_unique[cpp_hybrid_scan_reader](
                metadata.c_obj,
                options.c_obj
            )
        return reader

    def parquet_metadata(self):
        """Get the Parquet file footer metadata.

        Returns
        -------
        FileMetaData
            Parquet file footer metadata
        """
        cdef cpp_FileMetaData c_result
        with nogil:
            c_result = self.c_obj.get()[0].parquet_metadata()
        return c_FileMetaData.from_cpp(c_result)

    def page_index_byte_range(self):
        """Get the byte range of the page index.

        Returns
        -------
        ByteRangeInfo
            Byte range of the page index
        """
        cdef byte_range_info info
        with nogil:
            info = self.c_obj.get()[0].page_index_byte_range()
        return ByteRangeInfo(info.offset(), info.size())

    def setup_page_index(self, const uint8_t[::1] page_index_bytes):
        """Setup the page index within the Parquet file metadata.

        Parameters
        ----------
        page_index_bytes : Buffer
            Parquet page index buffer bytes
        """
        with nogil:
            self.c_obj.get()[0].setup_page_index(
                host_span[const_uint8_t](&page_index_bytes[0], len(page_index_bytes))
            )

    def all_row_groups(self, ParquetReaderOptions options):
        """Get all available row groups from the parquet file.

        Parameters
        ----------
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[int]
            List of row group indices
        """
        cdef vector[size_type] row_groups
        with nogil:
            row_groups = self.c_obj.get()[0].all_row_groups(options.c_obj)
        return list(row_groups)

    def total_rows_in_row_groups(self, list row_group_indices):
        """Get the total number of top-level rows in the row groups.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices

        Returns
        -------
        int
            Total number of top-level rows
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef size_type result
        with nogil:
            result = self.c_obj.get()[0].total_rows_in_row_groups(
                host_span[const_size_type](indices_vec.data(), indices_vec.size())
            )
        return result

    def reset_column_selection(self):
        """Reset the column selection state.

        Resets the internal column selection state forcing re-selection of columns in
        subsequent filter and read operations
        """
        with nogil:
            self.c_obj.get()[0].reset_column_selection()

    def read(
        self,
        SourceInfo source_info,
        list row_group_indices,
        ParquetReaderOptions options,
        const uint8_t[::1] page_index_bytes=None,
        bool use_stats_filter=True,
        bool use_dictionary_filter=True,
        bool use_bloom_filter=True,
        bool prune_filter_column_pages=False,
        bool prune_payload_column_pages=True,
        object stream=None,
        DeviceMemoryResource mr=None,
    ):
        """Read the Parquet file in a single fused hybrid scan operation.

        Performs row group pruning, row mask construction, and the two-pass
        filter/payload materialization in one C++ call, so the entire per-chunk
        read crosses into C++ (and acquires the GIL) once rather than once per
        step.

        Parameters
        ----------
        source_info : SourceInfo
            Source describing the single Parquet file backing this reader.
        row_group_indices : list[int]
            Candidate row group indices to read, before pruning.
        options : ParquetReaderOptions
            Parquet reader options, including the optional filter expression and
            column selection.
        page_index_bytes : memoryview, optional
            Host bytes of the page index. If ``None`` and the file contains a
            page index, the bytes are fetched from the source. Pass cached bytes
            to avoid a redundant read across reads of the same file.
        use_stats_filter : bool, default True
            Prune row groups using column chunk statistics.
        use_dictionary_filter : bool, default True
            Prune row groups using column chunk dictionary pages.
        use_bloom_filter : bool, default True
            Prune row groups using column chunk bloom filters.
        prune_filter_column_pages : bool, default False
            Whether to build and use a data page mask to prune filter column pages.
        prune_payload_column_pages : bool, default True
            Whether to build and use a data page mask to prune payload column pages.
        stream : Stream, optional
            CUDA stream.
        mr : DeviceMemoryResource, optional
            Device memory resource.

        Returns
        -------
        TableWithMetadata
            Materialized columns and metadata, in projection order.
        """
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)

        cdef vector[unique_ptr[datasource]] sources = make_datasources(source_info.c_obj)
        cdef vector[size_type] indices_vec = row_group_indices

        cdef cpp_hybrid_scan_read_options read_options
        read_options.use_stats_filter = use_stats_filter
        read_options.use_dictionary_filter = use_dictionary_filter
        read_options.use_bloom_filter = use_bloom_filter
        read_options.prune_filter_column_pages = (
            cpp_use_data_page_mask.YES if prune_filter_column_pages
            else cpp_use_data_page_mask.NO
        )
        read_options.prune_payload_column_pages = (
            cpp_use_data_page_mask.YES if prune_payload_column_pages
            else cpp_use_data_page_mask.NO
        )
        if page_index_bytes is not None and page_index_bytes.shape[0] > 0:
            read_options.page_index_bytes = host_span[const_uint8_t](
                &page_index_bytes[0], page_index_bytes.shape[0]
            )

        cdef datasource* source_ptr = sources[0].get()
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].read(
                dereference(source_ptr),
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                read_options,
                _stream.view().value(),
                mr.get_mr(),
            ))
        return TableWithMetadata.from_libcudf(c_result, _stream, mr)

    def filter_row_groups_with_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None
    ):
        """Filter row groups using column chunk statistics.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        list[int]
            Filtered row group indices
        """
        cdef Stream _stream = _get_stream(stream)
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[size_type] filtered
        with nogil:
            filtered = move(self.c_obj.get()[0].filter_row_groups_with_stats(
                host_span[const_size_type](
                    indices_vec.data(), indices_vec.size()
                ),
                options.c_obj,
                _stream.view().value()
            ))
        return list(filtered)

    def secondary_filters_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of bloom filters and dictionary pages.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        tuple[list[ByteRangeInfo], list[ByteRangeInfo]]
            Tuple of (bloom_filter_ranges, dictionary_page_ranges)
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef pair[vector[byte_range_info], vector[byte_range_info]] ranges
        with nogil:
            ranges = move(self.c_obj.get()[0].secondary_filters_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            ))

        bloom_ranges = [
            ByteRangeInfo(r.offset(), r.size()) for r in ranges.first
        ]
        dict_ranges = [
            ByteRangeInfo(r.offset(), r.size()) for r in ranges.second
        ]
        return (bloom_ranges, dict_ranges)

    def filter_row_groups_with_dictionary_pages(
        self,
        list dictionary_page_data,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None
    ):
        """Filter row groups using column chunk dictionary pages.

        Parameters
        ----------
        dictionary_page_data : list[Span]
            Span-like objects containing dictionary page data
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        list[int]
            Filtered row group indices
        """
        cdef vector[device_span[const_uint8_t]] spans_vec
        cdef Stream _stream = _get_stream(stream)
        for span in dictionary_page_data:
            spans_vec.push_back(_get_device_span(span))

        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[size_type] filtered
        with nogil:
            filtered = move(self.c_obj.get()[0].filter_row_groups_with_dictionary_pages(
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _stream.view().value()
            ))
        return list(filtered)

    def filter_row_groups_with_bloom_filters(
        self,
        list bloom_filter_data,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None
    ):
        """Filter row groups using column chunk bloom filters.

        Parameters
        ----------
        bloom_filter_data : list[Span]
            Span-like objects containing bloom filter data
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        list[int]
            Filtered row group indices
        """
        cdef vector[device_span[const_uint8_t]] spans_vec
        cdef Stream _stream = _get_stream(stream)
        for span in bloom_filter_data:
            spans_vec.push_back(_get_device_span(span))

        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[size_type] filtered
        with nogil:
            filtered = move(self.c_obj.get()[0].filter_row_groups_with_bloom_filters(
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _stream.view().value()
            ))
        return list(filtered)

    def build_all_true_row_mask(
        self,
        list row_group_indices,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Build an all-true boolean survival column for the given row groups.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        Column
            All-true boolean column with one entry per row across all row groups
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].build_all_true_row_mask(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                _stream.view().value(),
                mr.get_mr()
            ))
        return Column.from_libcudf(move(c_result), _stream, mr)

    def build_row_mask_with_page_index_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Build a boolean column indicating surviving rows from page stats.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        Column
            Boolean column indicating surviving rows
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        cdef unique_ptr[column] c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].build_row_mask_with_page_index_stats(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            ))
        return Column.from_libcudf(move(c_result), _stream, mr)

    def filter_column_chunks_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of column chunks of filter columns.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[ByteRangeInfo]
            Byte ranges to column chunks of filter columns
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[byte_range_info] ranges
        with nogil:
            ranges = move(self.c_obj.get()[0].filter_column_chunks_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            ))
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def materialize_filter_columns(
        self,
        list row_group_indices,
        list column_chunk_data,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize filter columns and update the row mask.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        column_chunk_data : list[Span]
            Span-like objects containing column chunk data of filter columns
        row_mask : Column
            Mutable boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        TableWithMetadata
            Table of materialized filter columns and metadata
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_span[const_uint8_t]] spans_vec
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))

        cdef mutable_column_view mask_view = row_mask.mutable_view()
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_filter_columns(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                mask_view,
                mask_data_pages,
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            ))
        return TableWithMetadata.from_libcudf(c_result, _stream, mr)

    def payload_column_chunks_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of column chunks of payload columns.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[ByteRangeInfo]
            Byte ranges to column chunks of payload columns
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[byte_range_info] ranges
        with nogil:
            ranges = move(self.c_obj.get()[0].payload_column_chunks_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            ))
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def materialize_payload_columns(
        self,
        list row_group_indices,
        list column_chunk_data,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize payload columns and apply the row mask.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        column_chunk_data : list[Span]
            Span-like objects containing column chunk data of payload columns
        row_mask : Column
            Boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        TableWithMetadata
            Table of materialized payload columns and metadata
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_span[const_uint8_t]] spans_vec
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))

        cdef column_view mask_view = row_mask.view()
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_payload_columns(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                mask_view,
                mask_data_pages,
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            ))
        return TableWithMetadata.from_libcudf(c_result, _stream, mr)

    def all_column_chunks_byte_ranges(
        self,
        list row_group_indices,
        ParquetReaderOptions options
    ):
        """Get byte ranges of column chunks of all columns.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        options : ParquetReaderOptions
            Parquet reader options

        Returns
        -------
        list[ByteRangeInfo]
            Byte ranges to column chunks of all columns
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[byte_range_info] ranges
        with nogil:
            ranges = move(self.c_obj.get()[0].all_column_chunks_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            ))
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def materialize_all_columns(
        self,
        list row_group_indices,
        list column_chunk_data,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize all columns.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        column_chunk_data : list[Span]
            Span-like objects containing column chunk data of all columns
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource

        Returns
        -------
        TableWithMetadata
            Table of materialized all columns and metadata
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_span[const_uint8_t]] spans_vec
        cdef Stream _stream = _get_stream(stream)
        mr = _get_memory_resource(mr)
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_all_columns(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                options.c_obj,
                _stream.view().value(),
                mr.get_mr()
            ))
        return TableWithMetadata.from_libcudf(c_result, _stream, mr)

    def setup_chunking_for_filter_columns(
        self,
        size_t chunk_read_limit,
        size_t pass_read_limit,
        list row_group_indices,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        list column_chunk_data,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Setup chunking information for filter columns.

        Parameters
        ----------
        chunk_read_limit : int
            Limit on bytes returned per chunk (0 for no limit)
        pass_read_limit : int
            Limit on memory for reading/decompressing (0 for no limit)
        row_group_indices : list[int]
            Input row group indices
        row_mask : Column
            Boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        column_chunk_data : list[Span]
            Span-like objects containing column chunk data of filter columns
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_span[const_uint8_t]] spans_vec
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))

        self._stream = _get_stream(stream)
        self.mr = _get_memory_resource(mr)

        cdef column_view mask_view = row_mask.view()
        with nogil:
            self.c_obj.get()[0].setup_chunking_for_filter_columns(
                chunk_read_limit,
                pass_read_limit,
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                mask_view,
                mask_data_pages,
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                options.c_obj,
                self._stream.view().value(),
                self.mr.get_mr()
            )

    def materialize_filter_columns_chunk(
        self,
        Column row_mask
    ):
        """Materialize a chunk of filter columns.

        Parameters
        ----------
        row_mask : Column
            Mutable boolean column indicating surviving rows
        Returns
        -------
        TableWithMetadata
            Table chunk of materialized filter columns and metadata
        """
        cdef mutable_column_view mask_view = row_mask.mutable_view()
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_filter_columns_chunk(
                mask_view
            ))
        return TableWithMetadata.from_libcudf(
            c_result, self._stream, self.mr
        )

    def setup_chunking_for_payload_columns(
        self,
        size_t chunk_read_limit,
        size_t pass_read_limit,
        list row_group_indices,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        list column_chunk_data,
        ParquetReaderOptions options,
        object stream=None,
        DeviceMemoryResource mr=None
    ):
        """Setup chunking information for payload columns.

        Parameters
        ----------
        chunk_read_limit : int
            Limit on bytes returned per chunk (0 for no limit)
        pass_read_limit : int
            Limit on memory for reading/decompressing (0 for no limit)
        row_group_indices : list[int]
            Input row group indices
        row_mask : Column
            Boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        column_chunk_data : list[Span]
            Span-like objects containing column chunk data of payload columns
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        mr : DeviceMemoryResource, optional
            Device memory resource
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_span[const_uint8_t]] spans_vec
        for span in column_chunk_data:
            spans_vec.push_back(_get_device_span(span))

        self._stream = _get_stream(stream)
        self.mr = _get_memory_resource(mr)

        cdef column_view mask_view = row_mask.view()
        with nogil:
            self.c_obj.get()[0].setup_chunking_for_payload_columns(
                chunk_read_limit,
                pass_read_limit,
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                mask_view,
                mask_data_pages,
                host_span[const_device_span_const_uint8_t](
                    <const_device_span_const_uint8_t*>spans_vec.data(), spans_vec.size()
                ),
                options.c_obj,
                self._stream.view().value(),
                self.mr.get_mr()
            )

    def materialize_payload_columns_chunk(
        self,
        Column row_mask,
    ):
        """Materialize a chunk of payload columns.

        Parameters
        ----------
        row_mask : Column
            Boolean column indicating surviving rows
        Returns
        -------
        TableWithMetadata
            Table chunk of materialized payload columns and metadata
        """
        cdef column_view mask_view = row_mask.view()
        cdef table_with_metadata c_result
        with nogil:
            c_result = move(self.c_obj.get()[0].materialize_payload_columns_chunk(
                mask_view
            ))
        return TableWithMetadata.from_libcudf(
            c_result, self._stream, self.mr
        )

    def construct_row_group_passes(
        self,
        list row_group_indices,
        size_t pass_read_limit,
    ):
        """Partition row groups into passes such that the GPU memory required to
        materialize a pass is bounded by the specified limit.

        Note that ``pass_read_limit`` is a hint, not an absolute limit. i.e. if
        a row group cannot fit within the limit, it will still constitute a valid
        pass.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        pass_read_limit : int
            Limit on the amount of memory used for reading and decompressing data
        or 0 if there is no limit.

        Returns
        -------
        list[list[int]]
            Lists of row group indices, one per pass.

        Raises
        ------
        ValueError
            If ``row_group_indices`` is empty.
        """
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[vector[size_type]] passes
        with nogil:
            passes = move(self.c_obj.get()[0].construct_row_group_passes(
                host_span[const_size_type](
                    indices_vec.data(), indices_vec.size()
                ),
                pass_read_limit
            ))
        return passes

    def has_next_table_chunk(self):
        """Check if there is any parquet data left to read.

        Returns
        -------
        bool
            True if there is data left to read
        """
        cdef bool result
        with nogil:
            result = self.c_obj.get()[0].has_next_table_chunk()
        return result


UseDataPageMask.__str__ = UseDataPageMask.__repr__
