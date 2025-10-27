# Copyright (c) 2025, NVIDIA CORPORATION.

from libc.stdint cimport uint8_t
from libcpp.memory cimport make_unique, unique_ptr
from libcpp.pair cimport pair
from libcpp.utility cimport move
from libcpp.vector cimport vector

from rmm.librmm.device_buffer cimport device_buffer
from rmm.pylibrmm.device_buffer cimport DeviceBuffer
from rmm.pylibrmm.memory_resource cimport DeviceMemoryResource
from rmm.pylibrmm.stream cimport Stream

from pylibcudf.column cimport Column
from pylibcudf.io.parquet cimport ParquetReaderOptions
from pylibcudf.io.types cimport TableWithMetadata
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view, mutable_column_view
from pylibcudf.libcudf.io.hybrid_scan cimport (
    const_size_type,
    const_uint8_t,
    hybrid_scan_reader as cpp_hybrid_scan_reader,
    use_data_page_mask as cpp_use_data_page_mask,
)
from pylibcudf.libcudf.io.parquet_schema cimport FileMetaData as cpp_FileMetaData
from pylibcudf.libcudf.io.text cimport byte_range_info
from pylibcudf.libcudf.io.types cimport table_with_metadata
from pylibcudf.libcudf.types cimport size_type
from pylibcudf.libcudf.utilities.span cimport host_span
from pylibcudf.utils cimport _get_memory_resource, _get_stream

import pylibcudf.libcudf.io.hybrid_scan

UseDataPageMask = pylibcudf.libcudf.io.hybrid_scan.use_data_page_mask

__all__ = [
    "HybridScanReader",
    "UseDataPageMask",
    "FileMetaData",
    "ByteRangeInfo",
]


cdef class ByteRangeInfo:
    """Information about a byte range in a file.

    For details, see :cpp:class:`cudf::io::text::byte_range_info`

    Parameters
    ----------
    offset : int
        Offset in bytes from the start of the file
    size : int
        Size of the range in bytes
    """

    def __init__(self, size_t offset, size_t size):
        self.c_obj = byte_range_info(offset, size)

    @property
    def offset(self):
        """Get the offset in bytes."""
        return self.c_obj.offset()

    @property
    def size(self):
        """Get the size in bytes."""
        return self.c_obj.size()


cdef class FileMetaData:
    """Parquet file footer metadata.

    For details, see :cpp:class:`cudf::io::parquet::FileMetaData`
    """

    def __init__(self):
        raise ValueError("FileMetaData cannot be constructed directly")

    @staticmethod
    cdef FileMetaData from_cpp(cpp_FileMetaData metadata):
        cdef FileMetaData result = FileMetaData.__new__(FileMetaData)
        result.c_obj = metadata
        return result

    @property
    def version(self):
        """Get the file format version."""
        return self.c_obj.version

    @property
    def num_rows(self):
        """Get the total number of rows."""
        return self.c_obj.num_rows

    @property
    def created_by(self):
        """Get the application that created the file."""
        return self.c_obj.created_by.decode('utf-8')


cdef class HybridScanReader:
    """Experimental Parquet reader optimized for highly selective filters.

    This class implements a hybrid scan operation for reading Parquet files
    with highly selective filters. It reads in two passes: first reading
    filter columns to build a row mask, then reading payload columns using
    that mask for optimization.

    For details, see :cpp:class:`cudf::io::parquet::experimental::hybrid_scan_reader`

    Parameters
    ----------
    footer_bytes : bytes
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

    def __init__(self, bytes footer_bytes, ParquetReaderOptions options):
        cdef const uint8_t[::1] footer_view = footer_bytes
        self.c_obj = make_unique[cpp_hybrid_scan_reader](
            host_span[const_uint8_t](&footer_view[0], len(footer_bytes)),
            options.c_obj
        )

    @staticmethod
    def from_parquet_metadata(FileMetaData metadata, ParquetReaderOptions options):
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
        return FileMetaData.from_cpp(self.c_obj.get()[0].parquet_metadata())

    def page_index_byte_range(self):
        """Get the byte range of the page index.

        Returns
        -------
        ByteRangeInfo
            Byte range of the page index
        """
        cdef byte_range_info info = self.c_obj.get()[0].page_index_byte_range()
        return ByteRangeInfo(info.offset(), info.size())

    def setup_page_index(self, bytes page_index_bytes):
        """Setup the page index within the Parquet file metadata.

        Parameters
        ----------
        page_index_bytes : bytes
            Parquet page index buffer bytes
        """
        cdef const uint8_t[::1] page_view = page_index_bytes
        self.c_obj.get()[0].setup_page_index(
            host_span[const_uint8_t](&page_view[0], len(page_index_bytes))
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
        cdef vector[size_type] row_groups = self.c_obj.get()[0].all_row_groups(
            options.c_obj
        )
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
        return self.c_obj.get()[0].total_rows_in_row_groups(
            host_span[const_size_type](indices_vec.data(), indices_vec.size())
        )

    def filter_row_groups_with_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        Stream stream=None
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
        cdef vector[size_type] indices_vec = row_group_indices
        cdef vector[size_type] filtered = (
            self.c_obj.get()[0].filter_row_groups_with_stats(
                host_span[const_size_type](
                    indices_vec.data(), indices_vec.size()
                ),
                options.c_obj,
                _get_stream(stream).view()
            )
        )
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
        cdef pair[vector[byte_range_info], vector[byte_range_info]] ranges = \
            self.c_obj.get()[0].secondary_filters_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            )

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
        Stream stream=None
    ):
        """Filter row groups using column chunk dictionary pages.

        Parameters
        ----------
        dictionary_page_data : list[DeviceBuffer]
            Device buffers containing dictionary page data
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
        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in dictionary_page_data:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(None).get_mr()
            )

        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[size_type] filtered = \
            self.c_obj.get()[0].filter_row_groups_with_dictionary_pages(
                host_span[device_buffer](buffers_vec.data(), buffers_vec.size()),
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _get_stream(stream).view()
            )
        return list(filtered)

    def filter_row_groups_with_bloom_filters(
        self,
        list bloom_filter_data,
        list row_group_indices,
        ParquetReaderOptions options,
        Stream stream=None
    ):
        """Filter row groups using column chunk bloom filters.

        Parameters
        ----------
        bloom_filter_data : list[DeviceBuffer]
            Device buffers containing bloom filter data
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
        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in bloom_filter_data:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(None).get_mr()
            )

        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[size_type] filtered = \
            self.c_obj.get()[0].filter_row_groups_with_bloom_filters(
                host_span[device_buffer](buffers_vec.data(), buffers_vec.size()),
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _get_stream(stream).view()
            )
        return list(filtered)

    def build_row_mask_with_page_index_stats(
        self,
        list row_group_indices,
        ParquetReaderOptions options,
        Stream stream=None,
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
        cdef unique_ptr[column] c_result = \
            self.c_obj.get()[0].build_row_mask_with_page_index_stats(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj,
                _get_stream(stream).view(),
                _get_memory_resource(mr).get_mr()
            )
        return Column.from_libcudf(
            move(c_result), _get_stream(stream), _get_memory_resource(mr)
        )

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
        cdef vector[byte_range_info] ranges = \
            self.c_obj.get()[0].filter_column_chunks_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            )
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def materialize_filter_columns(
        self,
        list row_group_indices,
        list column_chunk_buffers,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        ParquetReaderOptions options,
        Stream stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize filter columns and update the row mask.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        column_chunk_buffers : list[DeviceBuffer]
            Device buffers containing column chunk data
        row_mask : Column
            Mutable boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        TableWithMetadata
            Table of materialized filter columns and metadata
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in column_chunk_buffers:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(mr).get_mr()
            )

        cdef mutable_column_view mask_view = row_mask.mutable_view()
        cdef table_with_metadata c_result = \
            self.c_obj.get()[0].materialize_filter_columns(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                move(buffers_vec),
                mask_view,
                mask_data_pages,
                options.c_obj,
                _get_stream(stream).view()
            )
        return TableWithMetadata.from_libcudf(c_result, stream, mr)

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
        cdef vector[byte_range_info] ranges = \
            self.c_obj.get()[0].payload_column_chunks_byte_ranges(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                options.c_obj
            )
        return [ByteRangeInfo(r.offset(), r.size()) for r in ranges]

    def materialize_payload_columns(
        self,
        list row_group_indices,
        list column_chunk_buffers,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        ParquetReaderOptions options,
        Stream stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize payload columns and apply the row mask.

        Parameters
        ----------
        row_group_indices : list[int]
            Input row group indices
        column_chunk_buffers : list[DeviceBuffer]
            Device buffers containing column chunk data
        row_mask : Column
            Boolean column indicating surviving rows
        mask_data_pages : UseDataPageMask
            Whether to use a data page mask
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        TableWithMetadata
            Table of materialized payload columns and metadata
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in column_chunk_buffers:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(mr).get_mr()
            )

        cdef column_view mask_view = row_mask.view()
        cdef table_with_metadata c_result = \
            self.c_obj.get()[0].materialize_payload_columns(
                host_span[const_size_type](indices_vec.data(), indices_vec.size()),
                move(buffers_vec),
                mask_view,
                mask_data_pages,
                options.c_obj,
                _get_stream(stream).view()
            )
        return TableWithMetadata.from_libcudf(c_result, stream, mr)

    def setup_chunking_for_filter_columns(
        self,
        size_t chunk_read_limit,
        size_t pass_read_limit,
        list row_group_indices,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        list column_chunk_buffers,
        ParquetReaderOptions options,
        Stream stream=None
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
        column_chunk_buffers : list[DeviceBuffer]
            Device buffers containing column chunk data
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in column_chunk_buffers:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(None).get_mr()
            )

        cdef column_view mask_view = row_mask.view()
        self.c_obj.get()[0].setup_chunking_for_filter_columns(
            chunk_read_limit,
            pass_read_limit,
            host_span[const_size_type](indices_vec.data(), indices_vec.size()),
            mask_view,
            mask_data_pages,
            move(buffers_vec),
            options.c_obj,
            _get_stream(stream).view()
        )

    def materialize_filter_columns_chunk(
        self,
        Column row_mask,
        Stream stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize a chunk of filter columns.

        Parameters
        ----------
        row_mask : Column
            Mutable boolean column indicating surviving rows
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        TableWithMetadata
            Table chunk of materialized filter columns and metadata
        """
        cdef mutable_column_view mask_view = row_mask.mutable_view()
        cdef table_with_metadata c_result = \
            self.c_obj.get()[0].materialize_filter_columns_chunk(
                mask_view,
                _get_stream(stream).view()
            )
        return TableWithMetadata.from_libcudf(c_result, stream, mr)

    def setup_chunking_for_payload_columns(
        self,
        size_t chunk_read_limit,
        size_t pass_read_limit,
        list row_group_indices,
        Column row_mask,
        cpp_use_data_page_mask mask_data_pages,
        list column_chunk_buffers,
        ParquetReaderOptions options,
        Stream stream=None
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
        column_chunk_buffers : list[DeviceBuffer]
            Device buffers containing column chunk data
        options : ParquetReaderOptions
            Parquet reader options
        stream : Stream, optional
            CUDA stream
        """
        cdef vector[size_type] indices_vec = row_group_indices

        cdef vector[device_buffer] buffers_vec
        cdef DeviceBuffer dev_buf
        cdef const device_buffer* c_dev_buf_ptr
        for buf in column_chunk_buffers:
            dev_buf = <DeviceBuffer>buf
            c_dev_buf_ptr = dev_buf.c_obj.get()
            # Create a new device_buffer by copying from the source device buffer
            buffers_vec.emplace_back(
                c_dev_buf_ptr.data(),  # source device pointer
                c_dev_buf_ptr.size(),  # size in bytes
                _get_stream(stream).view(),
                _get_memory_resource(None).get_mr()
            )

        cdef column_view mask_view = row_mask.view()
        self.c_obj.get()[0].setup_chunking_for_payload_columns(
            chunk_read_limit,
            pass_read_limit,
            host_span[const_size_type](indices_vec.data(), indices_vec.size()),
            mask_view,
            mask_data_pages,
            move(buffers_vec),
            options.c_obj,
            _get_stream(stream).view()
        )

    def materialize_payload_columns_chunk(
        self,
        Column row_mask,
        Stream stream=None,
        DeviceMemoryResource mr=None
    ):
        """Materialize a chunk of payload columns.

        Parameters
        ----------
        row_mask : Column
            Boolean column indicating surviving rows
        stream : Stream, optional
            CUDA stream

        Returns
        -------
        TableWithMetadata
            Table chunk of materialized payload columns and metadata
        """
        cdef column_view mask_view = row_mask.view()
        cdef table_with_metadata c_result = \
            self.c_obj.get()[0].materialize_payload_columns_chunk(
                mask_view,
                _get_stream(stream).view()
            )
        return TableWithMetadata.from_libcudf(c_result, stream, mr)

    def has_next_table_chunk(self):
        """Check if there is any parquet data left to read.

        Returns
        -------
        bool
            True if there is data left to read
        """
        return self.c_obj.get()[0].has_next_table_chunk()


UseDataPageMask.__str__ = UseDataPageMask.__repr__
