# Copyright (c) 2024, NVIDIA CORPORATION.
from pylibcudf.io.types import SourceInfo, TableWithMetadata

__all__ = ["read_avro"]

def read_avro(
    source_info: SourceInfo,
    columns: list[str] | None = None,
    skip_rows: int = 0,
    num_rows: int = -1,
) -> TableWithMetadata: ...
