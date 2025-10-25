# Copyright (c) 2025, NVIDIA CORPORATION.
import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rmm import DeviceBuffer
from rmm.pylibrmm.stream import Stream

import pylibcudf as plc
from pylibcudf.expressions import (
    ASTOperator,
    ColumnNameReference,
    Literal,
    Operation,
)
from pylibcudf.io.experimental import HybridScanReader, UseDataPageMask

# Shared kwargs to pass to make_source
_COMMON_PARQUET_SOURCE_KWARGS = {"format": "parquet"}


@pytest.fixture
def simple_parquet_table():
    """Create a simple PyArrow table for testing."""
    data = {
        "col0": pa.array(list(range(1000)), type=pa.uint32()),
        "col1": [f"str_{i}" for i in range(1000)],
        "col2": [float(i) * 1.5 for i in range(1000)],
    }
    return pa.table(data)


@pytest.fixture
def simple_parquet_bytes(simple_parquet_table):
    """Create parquet bytes from the simple table."""
    buf = io.BytesIO()
    pq.write_table(
        simple_parquet_table,
        buf,
        row_group_size=250,
        compression="SNAPPY",
        use_dictionary=True,
        write_statistics=True,
    )
    return buf.getvalue()


def extract_footer(parquet_bytes):
    """Extract footer bytes from a parquet file."""
    footer_size = int.from_bytes(parquet_bytes[-8:-4], byteorder="little")
    footer_start = len(parquet_bytes) - 8 - footer_size
    return parquet_bytes[footer_start:-8]


def test_hybrid_scan_reader_basic(simple_parquet_bytes):
    """Test basic HybridScanReader construction and metadata access."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    # Create reader options
    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    # Create reader
    reader = HybridScanReader(footer_bytes, options)

    # Test metadata access
    metadata = reader.parquet_metadata()
    assert metadata.version == 2
    assert metadata.num_rows == 1000
    assert "parquet-cpp-arrow" in metadata.created_by


def test_hybrid_scan_reader_from_metadata(simple_parquet_bytes):
    """Test creating HybridScanReader from pre-populated metadata."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    # First create a reader to get metadata
    reader1 = HybridScanReader(footer_bytes, options)
    metadata = reader1.parquet_metadata()

    # Create a second reader from the metadata
    reader2 = HybridScanReader.from_parquet_metadata(metadata, options)

    # Verify both readers work the same
    assert (
        reader1.parquet_metadata().num_rows
        == reader2.parquet_metadata().num_rows
    )


def test_hybrid_scan_all_row_groups(simple_parquet_bytes):
    """Test getting all row groups."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    reader = HybridScanReader(footer_bytes, options)
    row_groups = reader.all_row_groups(options)

    # We created 4 row groups (1000 rows / 250 row_group_size)
    assert len(row_groups) == 4
    assert row_groups == [0, 1, 2, 3]


def test_hybrid_scan_total_rows_in_row_groups(simple_parquet_bytes):
    """Test getting total rows in specific row groups."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    reader = HybridScanReader(footer_bytes, options)
    all_row_groups = reader.all_row_groups(options)

    # Test all row groups
    total_rows = reader.total_rows_in_row_groups(all_row_groups)
    assert total_rows == 1000

    # Test subset of row groups
    subset_rows = reader.total_rows_in_row_groups([0, 1])
    assert subset_rows == 500


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_filter_row_groups_with_stats(
    simple_parquet_bytes, stream
):
    """Test filtering row groups using statistics."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    # Filter: col0 >= 500
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(500, type=pa.uint32()))),
    )
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)
    all_row_groups = reader.all_row_groups(options)
    filtered = reader.filter_row_groups_with_stats(
        all_row_groups, options, stream
    )

    # Row groups 0-1 have rows 0-499, should be filtered out
    # Row groups 2-3 have rows 500-999, should remain
    assert filtered == [2, 3]


def test_hybrid_scan_page_index_byte_range(simple_parquet_bytes):
    """Test getting page index byte range."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    reader = HybridScanReader(footer_bytes, options)
    page_index_range = reader.page_index_byte_range()

    # PyArrow doesn't write page index by default
    assert page_index_range.offset == 0
    assert page_index_range.size == 0


def test_hybrid_scan_secondary_filters_byte_ranges(simple_parquet_bytes):
    """Test getting bloom filter and dictionary page byte ranges."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    # Need to set a filter for secondary filters to work
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(100, type=pa.uint32()))),
    )
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)
    all_row_groups = reader.all_row_groups(options)

    bloom_ranges, dict_ranges = reader.secondary_filters_byte_ranges(
        all_row_groups, options
    )

    # These should be lists of ByteRangeInfo
    assert isinstance(bloom_ranges, list)
    assert isinstance(dict_ranges, list)


def test_hybrid_scan_column_chunk_byte_ranges(simple_parquet_bytes):
    """Test getting filter and payload column chunk byte ranges."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()

    # Set filter to make col0 the filter column
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(100, type=pa.uint32()))),
    )
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)
    all_row_groups = reader.all_row_groups(options)

    # Get filter column ranges
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        all_row_groups, options
    )
    assert len(filter_ranges) == 4  # One per row group

    # Get payload column ranges
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        all_row_groups, options
    )
    assert len(payload_ranges) == 8  # Two columns * 4 row groups

    # Verify all have valid offsets and sizes
    for r in filter_ranges + payload_ranges:
        assert r.offset >= 0
        assert r.size > 0


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize(
    "use_data_page_mask", [UseDataPageMask.NO, UseDataPageMask.YES]
)
def test_hybrid_scan_materialize_columns(
    simple_parquet_bytes, simple_parquet_table, stream, use_data_page_mask
):
    """Test full workflow of materializing filter and payload columns."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    # Create filter: col0 >= 100
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(100, type=pa.uint32()))),
    )

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)

    # Get filtered row groups
    all_row_groups = reader.all_row_groups(options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, options, stream
    )

    # Create initial row mask
    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get filter column data
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, options
    )
    filter_buffers = [
        DeviceBuffer.to_device(
            simple_parquet_bytes[r.offset : r.offset + r.size]
        )
        for r in filter_ranges
    ]

    # Materialize filter columns (mr is optional, defaults to None)
    filter_result = reader.materialize_filter_columns(
        filtered_row_groups,
        filter_buffers,
        row_mask,
        use_data_page_mask,
        options,
        stream,
        None,  # mr parameter
    )

    assert filter_result.tbl.num_columns() == 1
    assert filter_result.tbl.num_rows() == 900  # 1000 - 100 filtered out

    # Get payload column data
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        filtered_row_groups, options
    )
    payload_buffers = [
        DeviceBuffer.to_device(
            simple_parquet_bytes[r.offset : r.offset + r.size]
        )
        for r in payload_ranges
    ]

    # Materialize payload columns (mr is optional, defaults to None)
    payload_result = reader.materialize_payload_columns(
        filtered_row_groups,
        payload_buffers,
        row_mask,
        use_data_page_mask,
        options,
        stream,
        None,  # mr parameter
    )

    assert payload_result.tbl.num_columns() == 2
    assert payload_result.tbl.num_rows() == 900

    # Verify results match regular parquet reader
    # Create a fresh BytesIO to avoid buffer position issues
    comparison_buffer = io.BytesIO(simple_parquet_bytes)
    comparison_source = plc.io.SourceInfo([comparison_buffer])
    comparison_options = plc.io.parquet.ParquetReaderOptions.builder(
        comparison_source
    ).build()
    comparison_options.set_filter(filter_expression)
    expected_result = plc.io.parquet.read_parquet(comparison_options, stream)

    # Combine hybrid scan results
    hybrid_columns = filter_result.tbl.columns() + payload_result.tbl.columns()
    hybrid_table = plc.Table(hybrid_columns)

    # Compare just the table data (metadata structures are complex)
    expected_arrow = expected_result.tbl.to_arrow()
    hybrid_arrow = hybrid_table.to_arrow()

    assert expected_arrow.equals(hybrid_arrow)


def test_hybrid_scan_has_next_table_chunk(simple_parquet_bytes):
    """Test has_next_table_chunk method - requires chunking to be set up first."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(100, type=pa.uint32()))),
    )

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)

    all_row_groups = reader.all_row_groups(options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, options
    )

    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, options
    )
    filter_buffers = [
        DeviceBuffer.to_device(
            simple_parquet_bytes[r.offset : r.offset + r.size]
        )
        for r in filter_ranges
    ]

    # Setup chunking first
    reader.setup_chunking_for_filter_columns(
        512,  # chunk_read_limit
        0,  # pass_read_limit
        filtered_row_groups,
        row_mask,
        UseDataPageMask.NO,
        filter_buffers,
        options,
    )

    # Now has_next_table_chunk should work
    has_next = reader.has_next_table_chunk()
    assert isinstance(has_next, bool)


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_chunked_reading(simple_parquet_bytes, stream):
    """Test chunked reading with setup and chunk methods."""
    footer_bytes = extract_footer(simple_parquet_bytes)

    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(plc.Scalar.from_arrow(pa.scalar(100, type=pa.uint32()))),
    )

    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    options = plc.io.parquet.ParquetReaderOptions.builder(source).build()
    options.set_filter(filter_expression)

    reader = HybridScanReader(footer_bytes, options)

    all_row_groups = reader.all_row_groups(options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, options, stream
    )

    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get column data
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, options
    )
    filter_buffers = [
        DeviceBuffer.to_device(
            simple_parquet_bytes[r.offset : r.offset + r.size]
        )
        for r in filter_ranges
    ]

    # Setup chunking for filter columns with small chunk size
    chunk_read_limit = 512  # Small limit to force multiple chunks
    pass_read_limit = 0  # No limit

    reader.setup_chunking_for_filter_columns(
        chunk_read_limit,
        pass_read_limit,
        filtered_row_groups,
        row_mask,
        UseDataPageMask.NO,
        filter_buffers,
        options,
        stream,
    )

    # Read chunks
    chunks_read = 0
    while reader.has_next_table_chunk():
        chunk_result = reader.materialize_filter_columns_chunk(
            row_mask, stream, None
        )
        assert isinstance(chunk_result, plc.io.types.TableWithMetadata)
        chunks_read += 1
        # Limit iterations to avoid infinite loop if something goes wrong
        if chunks_read > 100:
            break

    # We should have read at least one chunk
    assert chunks_read > 0
