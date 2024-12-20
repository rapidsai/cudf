# Copyright (c) 2024, NVIDIA CORPORATION.

from typing import Any

from pylibcudf.io.types import SourceInfo, TableWithMetadata
from pylibcudf.types import DataType

def read_orc(
    source_info: SourceInfo,
    columns: list[str] | None = None,
    stripes: list[list[int]] | None = None,
    skip_rows: int = 0,
    nrows: int = -1,
    use_index: bool = True,
    use_np_dtypes: bool = True,
    timestamp_type: DataType | None = None,
    decimal128_columns: list[str] | None = None,
) -> TableWithMetadata: ...

class OrcColumnStatistics:
    def __init__(self): ...
    @property
    def number_of_values(self) -> int | None: ...
    @property
    def has_null(self) -> bool | None: ...
    def __getitem__(self, item: str) -> Any: ...
    def __contains__(self, item: str) -> bool: ...
    def get[T](self, item: str, default: None | T = None) -> T | None: ...

class ParsedOrcStatistics:
    def __init__(self): ...
    @property
    def column_names(self) -> list[str]: ...
    @property
    def file_stats(self) -> list[OrcColumnStatistics]: ...
    @property
    def stripes_stats(self) -> list[OrcColumnStatistics]: ...

def read_parsed_orc_statistics(
    source_info: SourceInfo,
) -> ParsedOrcStatistics: ...
