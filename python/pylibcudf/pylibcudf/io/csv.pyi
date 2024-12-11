# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Mapping

from pylibcudf.io.types import (
    CompressionType,
    QuoteStyle,
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
)
from pylibcudf.table import Table
from pylibcudf.types import DataType

def read_csv(
    source_info: SourceInfo,
    *,
    compression: CompressionType = CompressionType.AUTO,
    byte_range_offset: int = 0,
    byte_range_size: int = 0,
    col_names: list[str] | None = None,
    prefix: str = "",
    mangle_dupe_cols: bool = True,
    usecols: list[int] | list[str] | None = None,
    nrows: int = -1,
    skiprows: int = 0,
    skipfooter: int = 0,
    header: int = 0,
    lineterminator: str = "\n",
    delimiter: str | None = None,
    thousands: str | None = None,
    decimal: str = ".",
    comment: str | None = None,
    delim_whitespace: bool = False,
    skipinitialspace: bool = False,
    skip_blank_lines: bool = True,
    quoting: QuoteStyle = QuoteStyle.MINIMAL,
    quotechar: str = '"',
    doublequote: bool = True,
    parse_dates: list[str] | list[int] | None = None,
    parse_hex: list[str] | list[int] | None = None,
    # Technically this should be dict/list
    # but using a fused type prevents using None as default
    dtypes: Mapping[str, DataType] | list[DataType] | None = None,
    true_values: list[str] | None = None,
    false_values: list[str] | None = None,
    na_values: list[str] | None = None,
    keep_default_na: bool = True,
    na_filter: bool = True,
    dayfirst: bool = False,
    # Note: These options are supported by the libcudf reader
    # but are not exposed here since there is no demand for them
    # on the Python side yet.
    # detect_whitespace_around_quotes: bool = False,
    # timestamp_type: DataType = DataType(type_id.EMPTY),
) -> TableWithMetadata: ...
def write_csv(options: CsvWriterOptionsBuilder) -> None: ...

class CsvWriterOptions:
    def __init__(self): ...
    @staticmethod
    def builder(sink: SinkInfo, table: Table) -> CsvWriterOptionsBuilder: ...

class CsvWriterOptionsBuilder:
    def __init__(self): ...
    def names(self, names: list) -> CsvWriterOptionsBuilder: ...
    def na_rep(self, val: str) -> CsvWriterOptionsBuilder: ...
    def include_header(self, val: bool) -> CsvWriterOptionsBuilder: ...
    def rows_per_chunk(self, val: int) -> CsvWriterOptionsBuilder: ...
    def line_terminator(self, term: str) -> CsvWriterOptionsBuilder: ...
    def inter_column_delimiter(
        self, delim: str
    ) -> CsvWriterOptionsBuilder: ...
    def true_value(self, val: str) -> CsvWriterOptionsBuilder: ...
    def false_value(self, val: str) -> CsvWriterOptionsBuilder: ...
    def build(self) -> CsvWriterOptions: ...
