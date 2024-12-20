# Copyright (c) 2024, NVIDIA CORPORATION.
from collections.abc import Mapping
from typing import TypeAlias

from pylibcudf.column import Column
from pylibcudf.io.types import (
    CompressionType,
    JSONRecoveryMode,
    SinkInfo,
    SourceInfo,
    TableWithMetadata,
)
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
def write_json(
    sink_info: SinkInfo,
    table_w_meta: TableWithMetadata,
    na_rep: str = "",
    include_nulls: bool = False,
    lines: bool = False,
    rows_per_chunk: int = 2**32 - 1,
    true_value: str = "true",
    false_value: str = "false",
) -> None: ...
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
