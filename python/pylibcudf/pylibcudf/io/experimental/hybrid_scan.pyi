# Copyright (c) 2025, NVIDIA CORPORATION.

from enum import IntEnum

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.io.parquet import ParquetReaderOptions
from pylibcudf.io.types import TableWithMetadata

class UseDataPageMask(IntEnum):
    YES: int
    NO: int

class ByteRangeInfo:
    def __init__(self, offset: int, size: int) -> None: ...
    @property
    def offset(self) -> int: ...
    @property
    def size(self) -> int: ...

class FileMetaData:
    @property
    def version(self) -> int: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def created_by(self) -> str: ...

class HybridScanReader:
    def __init__(
        self, footer_bytes: bytes, options: ParquetReaderOptions
    ) -> None: ...
    @staticmethod
    def from_parquet_metadata(
        metadata: FileMetaData, options: ParquetReaderOptions
    ) -> HybridScanReader: ...
    def parquet_metadata(self) -> FileMetaData: ...
    def page_index_byte_range(self) -> ByteRangeInfo: ...
    def setup_page_index(self, page_index_bytes: bytes) -> None: ...
    def all_row_groups(self, options: ParquetReaderOptions) -> list[int]: ...
    def total_rows_in_row_groups(
        self, row_group_indices: list[int]
    ) -> int: ...
    def filter_row_groups_with_stats(
        self,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> list[int]: ...
    def secondary_filters_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> tuple[list[ByteRangeInfo], list[ByteRangeInfo]]: ...
    def filter_row_groups_with_dictionary_pages(
        self,
        dictionary_page_data: list[DeviceBuffer],
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> list[int]: ...
    def filter_row_groups_with_bloom_filters(
        self,
        bloom_filter_data: list[DeviceBuffer],
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> list[int]: ...
    def build_row_mask_with_page_index_stats(
        self,
        row_group_indices: list[int],
        options: ParquetReaderOptions,
        stream: Stream = None,
        mr: DeviceMemoryResource = None,
    ) -> Column: ...
    def filter_column_chunks_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> list[ByteRangeInfo]: ...
    def materialize_filter_columns(
        self,
        row_group_indices: list[int],
        column_chunk_buffers: list[DeviceBuffer],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> TableWithMetadata: ...
    def payload_column_chunks_byte_ranges(
        self, row_group_indices: list[int], options: ParquetReaderOptions
    ) -> list[ByteRangeInfo]: ...
    def materialize_payload_columns(
        self,
        row_group_indices: list[int],
        column_chunk_buffers: list[DeviceBuffer],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> TableWithMetadata: ...
    def setup_chunking_for_filter_columns(
        self,
        chunk_read_limit: int,
        pass_read_limit: int,
        row_group_indices: list[int],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        column_chunk_buffers: list[DeviceBuffer],
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> None: ...
    def materialize_filter_columns_chunk(
        self, row_mask: Column, stream: Stream = None
    ) -> TableWithMetadata: ...
    def setup_chunking_for_payload_columns(
        self,
        chunk_read_limit: int,
        pass_read_limit: int,
        row_group_indices: list[int],
        row_mask: Column,
        mask_data_pages: UseDataPageMask,
        column_chunk_buffers: list[DeviceBuffer],
        options: ParquetReaderOptions,
        stream: Stream = None,
    ) -> None: ...
    def materialize_payload_columns_chunk(
        self, row_mask: Column, stream: Stream = None
    ) -> TableWithMetadata: ...
    def has_next_table_chunk(self) -> bool: ...
