# Copyright (c) 2024, NVIDIA CORPORATION.
import io
import os
from collections.abc import Mapping
from enum import IntEnum, auto
from typing import Literal, TypeAlias, overload

from pylibcudf.column import Column
from pylibcudf.io.datasource import Datasource
from pylibcudf.table import Table

class JSONRecoveryMode(IntEnum):
    FAIL = auto()
    RECOVER_WITH_NULL = auto()

class CompressionType(IntEnum):
    NONE = auto()
    AUTO = auto()
    SNAPPY = auto()
    GZIP = auto()
    BZIP2 = auto()
    BROTLI = auto()
    ZIP = auto()
    XZ = auto()
    ZLIB = auto()
    LZ4 = auto()
    LZO = auto()
    ZSTD = auto()

class ColumnEncoding(IntEnum):
    USE_DEFAULT = auto()
    DICTIONARY = auto()
    PLAIN = auto()
    DELTA_BINARY_PACKED = auto()
    DELTA_LENGTH_BYTE_ARRAY = auto()
    DELTA_BYTE_ARRAY = auto()
    BYTE_STREAM_SPLIT = auto()
    DIRECT = auto()
    DIRECT_V2 = auto()
    DICTIONARY_V2 = auto()

class DictionaryPolicy(IntEnum):
    NEVER = auto()
    ADAPTIVE = auto()
    ALWAYS = auto()

class StatisticsFreq(IntEnum):
    STATISTICS_NONE = auto()
    STATISTICS_ROWGROUP = auto()
    STATISTICS_PAGE = auto()
    STATISTICS_COLUMN = auto()

class QuoteStyle(IntEnum):
    MINIMAL = auto()
    ALL = auto()
    NONNUMERIC = auto()
    NONE = auto()

ColumnNameSpec: TypeAlias = tuple[str, list[ColumnNameSpec]]
ChildNameSpec: TypeAlias = Mapping[str, ChildNameSpec]

class TableWithMetadata:
    tbl: Table
    def __init__(
        self, tbl: Table, column_names: list[ColumnNameSpec]
    ) -> None: ...
    @property
    def columns(self) -> list[Column]: ...
    @overload
    def column_names(self, include_children: Literal[False]) -> list[str]: ...
    @overload
    def column_names(
        self, include_children: Literal[True]
    ) -> list[ColumnNameSpec]: ...
    @overload
    def column_names(
        self, include_children: bool = False
    ) -> list[str] | list[ColumnNameSpec]: ...
    @property
    def child_names(self) -> ChildNameSpec: ...
    @property
    def per_file_user_data(self) -> list[Mapping[str, str]]: ...

class SourceInfo:
    def __init__(
        self, sources: list[str] | list[os.PathLike] | list[Datasource]
    ) -> None: ...

class SinkInfo:
    def __init__(
        self,
        sinks: list[os.PathLike]
        | list[io.StringIO]
        | list[io.BytesIO]
        | list[io.TextIOBase]
        | list[str],
    ) -> None: ...
