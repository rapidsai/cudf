# Copyright (c) 2024, NVIDIA CORPORATION.
from collections.abc import Mapping
from typing import TypeAlias

from typing_extensions import Self

from pylibcudf.column import Column
from pylibcudf.io.types import (
    CompressionType,
    JSONRecoveryMode,
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
)
from pylibcudf.table import Table
from pylibcudf.types import DataType

ChildNameToTypeMap: TypeAlias = Mapping[str, ChildNameToTypeMap]

NameAndType: TypeAlias = tuple[str, DataType, list[NameAndType]]

def read_json(
    source_info: SourceInfo,
    dtypes: list[NameAndType] | None = None,
    compression: CompressionType = CompressionType.AUTO,
    lines: bool = False,
    byte_range_offset: int = 0,
    byte_range_size: int = 0,
    keep_quotes: bool = False,
    mixed_types_as_string: bool = False,
    prune_columns: bool = False,
    recovery_mode: JSONRecoveryMode = JSONRecoveryMode.FAIL,
) -> TableWithMetadata: ...

class JsonWriterOptions:
    @staticmethod
    def builder(sink: SinkInfo, table: Table) -> JsonWriterOptionsBuilder: ...
    def set_rows_per_chunk(self, val: int) -> None: ...
    def set_true_value(self, val: str) -> None: ...
    def set_false_value(self, val: str) -> None: ...

class JsonWriterOptionsBuilder:
    def metadata(self, tbl_w_meta: TableWithMetadata) -> Self: ...
    def na_rep(self, val: str) -> Self: ...
    def include_nulls(self, val: bool) -> Self: ...
    def lines(self, val: bool) -> Self: ...
    def build(self) -> JsonWriterOptions: ...

def write_json(options: JsonWriterOptions) -> None: ...
def chunked_read_json(
    source_info: SourceInfo,
    dtypes: list[NameAndType] | None = None,
    compression: CompressionType = CompressionType.AUTO,
    keep_quotes: bool = False,
    mixed_types_as_string: bool = False,
    prune_columns: bool = False,
    recovery_mode: JSONRecoveryMode = JSONRecoveryMode.FAIL,
    chunk_size: int = 100_000_000,
) -> tuple[list[Column], list[str], ChildNameToTypeMap]: ...
