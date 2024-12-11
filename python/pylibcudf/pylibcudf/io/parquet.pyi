# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.expressions import Expression
from pylibcudf.io.types import SourceInfo, TableWithMetadata

class ChunkedParquetReader:
    def __init__(
        self,
        source_info: SourceInfo,
        columns: list[str] | None = None,
        row_groups: list[list[int]] | None = None,
        use_pandas_metadata: bool = True,
        convert_strings_to_categories: bool = False,
        skip_rows: int = 0,
        nrows: int = 0,
        chunk_read_limit: int = 0,
        pass_read_limit: int = 1024000000,
        allow_mismatched_pq_schemas: bool = False,
    ) -> None: ...
    def has_next(self) -> bool: ...
    def read_chunk(self) -> TableWithMetadata: ...

def read_parquet(
    source_info: SourceInfo,
    columns: list[str] | None = None,
    row_groups: list[list[int]] | None = None,
    filters: Expression | None = None,
    convert_strings_to_categories: bool = False,
    use_pandas_metadata: bool = True,
    skip_rows: int = 0,
    nrows: int = -1,
    allow_mismatched_pq_schemas: bool = False,
    # disabled see comment in parquet.pyx for more
    # reader_column_schema: ReaderColumnSchema = *,
    # timestamp_type: DataType = *
) -> TableWithMetadata: ...
