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


@pytest.fixture(scope="module")
def num_rows():
    """Number of rows in the test table."""
    return 1000


@pytest.fixture(scope="module")
def row_group_size():
    """Row group size for parquet files."""
    return 250


@pytest.fixture(scope="module")
def num_row_groups(num_rows, row_group_size):
    """Number of row groups in the test parquet file."""
    return num_rows // row_group_size


@pytest.fixture(scope="module")
def simple_parquet_table(num_rows):
    """Create a simple PyArrow table for testing."""
    data = {
        "col0": pa.array(list(range(num_rows)), type=pa.uint32()),
        "col1": [f"str_{i}" for i in range(num_rows)],
        "col2": [float(i) * 1.5 for i in range(num_rows)],
    }
    return pa.table(data)


@pytest.fixture(scope="module")
def simple_parquet_bytes(simple_parquet_table, row_group_size):
    """Create parquet bytes from the simple table."""
    buf = io.BytesIO()
    pq.write_table(
        simple_parquet_table,
        buf,
        row_group_size=row_group_size,
        compression="SNAPPY",
        use_dictionary=True,
        write_statistics=True,
    )
    return buf.getvalue()


@pytest.fixture(scope="module")
def simple_parquet_footer(simple_parquet_bytes):
    """Extract footer bytes from the simple parquet file.

    According to the Parquet file format specification, the last 8 bytes of a
    Parquet file contain:
    - 4 bytes: footer length (little-endian int32)
    - 4 bytes: magic number "PAR1"

    The footer metadata itself is located immediately before these last 8 bytes.

    See: https://parquet.apache.org/docs/file-format/
    """
    # Parquet file format constants
    PARQUET_FOOTER_SIZE_BYTES = 4  # Number of bytes encoding footer length
    PARQUET_MAGIC_BYTES = 4  # Number of bytes for "PAR1" magic number
    PARQUET_SUFFIX_BYTES = (
        PARQUET_FOOTER_SIZE_BYTES + PARQUET_MAGIC_BYTES
    )  # Total: 8 bytes

    # Extract footer length from bytes [-8:-4]
    footer_size = int.from_bytes(
        simple_parquet_bytes[-PARQUET_SUFFIX_BYTES:-PARQUET_MAGIC_BYTES],
        byteorder="little",
    )

    # Footer is located before the 8-byte suffix
    footer_start = (
        len(simple_parquet_bytes) - PARQUET_SUFFIX_BYTES - footer_size
    )
    footer_end = len(simple_parquet_bytes) - PARQUET_SUFFIX_BYTES

    return simple_parquet_bytes[footer_start:footer_end]


@pytest.fixture
def simple_parquet_options(simple_parquet_bytes):
    """Create basic ParquetReaderOptions for the simple parquet file.

    Note: This is function-scoped (not module-scoped) because tests may
    modify the simple_parquet_options (e.g., by setting filters), so each test needs
    its own independent copy.
    """
    source = plc.io.SourceInfo([io.BytesIO(simple_parquet_bytes)])
    return plc.io.parquet.ParquetReaderOptions.builder(source).build()


def test_hybrid_scan_reader_basic(
    simple_parquet_footer, simple_parquet_options, num_rows
):
    """Test basic HybridScanReader construction and metadata access."""
    # Create reader
    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)

    # Test metadata access
    metadata = reader.parquet_metadata()
    assert metadata.version == 2
    assert metadata.num_rows == num_rows
    assert "parquet-cpp-arrow" in metadata.created_by


def test_hybrid_scan_reader_from_metadata(
    simple_parquet_footer, simple_parquet_options
):
    """Test creating HybridScanReader from pre-populated metadata."""

    # First create a reader to get metadata
    reader1 = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    metadata = reader1.parquet_metadata()

    # Create a second reader from the metadata
    reader2 = HybridScanReader.from_parquet_metadata(
        metadata, simple_parquet_options
    )

    # Verify both readers work the same
    assert (
        reader1.parquet_metadata().num_rows
        == reader2.parquet_metadata().num_rows
    )


def test_hybrid_scan_all_row_groups(
    simple_parquet_footer, simple_parquet_options, num_row_groups
):
    """Test getting all row groups."""

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    row_groups = reader.all_row_groups(simple_parquet_options)

    assert len(row_groups) == num_row_groups
    assert row_groups == list(range(num_row_groups))


def test_hybrid_scan_total_rows_in_row_groups(
    simple_parquet_footer, simple_parquet_options, num_rows, row_group_size
):
    """Test getting total rows in specific row groups."""

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    all_row_groups = reader.all_row_groups(simple_parquet_options)

    # Test all row groups
    total_rows = reader.total_rows_in_row_groups(all_row_groups)
    assert total_rows == num_rows

    # Test subset of row groups
    subset_rows = reader.total_rows_in_row_groups([0, 1])
    assert subset_rows == row_group_size * 2


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_filter_row_groups_with_stats(
    simple_parquet_footer,
    simple_parquet_options,
    num_row_groups,
    row_group_size,
    stream,
):
    """Test filtering row groups using statistics."""
    # Filter: col0 >= (num_row_groups // 2) * row_group_size
    # This should filter out the first half of row groups
    filter_threshold = (num_row_groups // 2) * row_group_size
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    all_row_groups = reader.all_row_groups(simple_parquet_options)
    filtered = reader.filter_row_groups_with_stats(
        all_row_groups, simple_parquet_options, stream
    )

    # First half of row groups should be filtered out, second half should remain
    expected_row_groups = list(range(num_row_groups // 2, num_row_groups))
    assert filtered == expected_row_groups


def test_hybrid_scan_page_index_byte_range(
    simple_parquet_footer, simple_parquet_options
):
    """Test getting page index byte range."""

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    page_index_range = reader.page_index_byte_range()

    # PyArrow doesn't write page index by default
    assert page_index_range.offset == 0
    assert page_index_range.size == 0


def test_hybrid_scan_secondary_filters_byte_ranges(
    simple_parquet_footer, simple_parquet_options, num_rows
):
    """Test getting bloom filter and dictionary page byte ranges."""

    # Need to set a filter for secondary filters to work
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    all_row_groups = reader.all_row_groups(simple_parquet_options)

    bloom_ranges, dict_ranges = reader.secondary_filters_byte_ranges(
        all_row_groups, simple_parquet_options
    )

    # These should be lists of ByteRangeInfo
    assert isinstance(bloom_ranges, list)
    assert isinstance(dict_ranges, list)


def test_hybrid_scan_column_chunk_byte_ranges(
    simple_parquet_footer, simple_parquet_options, num_rows, num_row_groups
):
    """Test getting filter and payload column chunk byte ranges."""

    # Set filter to make col0 the filter column
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )
    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)
    all_row_groups = reader.all_row_groups(simple_parquet_options)

    # Get filter column ranges (one filter column per row group)
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        all_row_groups, simple_parquet_options
    )
    assert len(filter_ranges) == num_row_groups

    # Get payload column ranges (two payload columns per row group)
    num_payload_columns = 2
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        all_row_groups, simple_parquet_options
    )
    assert len(payload_ranges) == num_payload_columns * num_row_groups

    # Verify all have valid offsets and sizes
    for r in filter_ranges + payload_ranges:
        assert r.offset >= 0
        assert r.size > 0


@pytest.mark.parametrize("stream", [None, Stream()])
@pytest.mark.parametrize(
    "use_data_page_mask", [UseDataPageMask.NO, UseDataPageMask.YES]
)
def test_hybrid_scan_materialize_columns(
    simple_parquet_bytes,
    simple_parquet_footer,
    simple_parquet_options,
    simple_parquet_table,
    num_rows,
    stream,
    use_data_page_mask,
):
    """Test full workflow of materializing filter and payload columns."""
    # Create filter: col0 >= num_rows // 10 (filter out first 10%)
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)

    # Get filtered row groups
    all_row_groups = reader.all_row_groups(simple_parquet_options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, simple_parquet_options, stream
    )

    # Create initial row mask
    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get filter column data
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
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
        simple_parquet_options,
        stream,
        None,  # mr parameter
    )

    # Filter column should have 1 column, with rows passing the filter
    expected_result_rows = num_rows - filter_threshold
    assert filter_result.tbl.num_columns() == 1
    assert filter_result.tbl.num_rows() == expected_result_rows

    # Get payload column data
    payload_ranges = reader.payload_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
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
        simple_parquet_options,
        stream,
        None,  # mr parameter
    )

    assert payload_result.tbl.num_columns() == 2
    assert payload_result.tbl.num_rows() == expected_result_rows

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


def test_hybrid_scan_has_next_table_chunk(
    simple_parquet_bytes,
    simple_parquet_footer,
    simple_parquet_options,
    num_rows,
):
    """Test has_next_table_chunk method - requires chunking to be set up first."""
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)

    all_row_groups = reader.all_row_groups(simple_parquet_options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, simple_parquet_options
    )

    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
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
        simple_parquet_options,
    )

    # Now has_next_table_chunk should work
    has_next = reader.has_next_table_chunk()
    assert isinstance(has_next, bool)


@pytest.mark.parametrize("stream", [None, Stream()])
def test_hybrid_scan_chunked_reading(
    simple_parquet_bytes,
    simple_parquet_footer,
    simple_parquet_options,
    num_rows,
    stream,
):
    """Test chunked reading with setup and chunk methods."""
    # Filter: col0 >= num_rows // 10
    filter_threshold = num_rows // 10
    filter_expression = Operation(
        ASTOperator.GREATER_EQUAL,
        ColumnNameReference("col0"),
        Literal(
            plc.Scalar.from_arrow(
                pa.scalar(filter_threshold, type=pa.uint32())
            )
        ),
    )

    simple_parquet_options.set_filter(filter_expression)

    reader = HybridScanReader(simple_parquet_footer, simple_parquet_options)

    all_row_groups = reader.all_row_groups(simple_parquet_options)
    filtered_row_groups = reader.filter_row_groups_with_stats(
        all_row_groups, simple_parquet_options, stream
    )

    total_rows = reader.total_rows_in_row_groups(filtered_row_groups)
    row_mask_array = pa.array([True] * total_rows, type=pa.bool_())
    row_mask = plc.Column.from_arrow(row_mask_array)

    # Get column data
    filter_ranges = reader.filter_column_chunks_byte_ranges(
        filtered_row_groups, simple_parquet_options
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
        simple_parquet_options,
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
