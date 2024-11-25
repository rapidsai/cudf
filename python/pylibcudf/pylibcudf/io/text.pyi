# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Sequence

from pylibcudf.column import Column

class ParseOptions:
    def __init__(
        self,
        *,
        byte_range: Sequence[int] | None = None,
        strip_delimiters: bool = False,
    ): ...

class DataChunkSource:
    def __init__(self, data: str): ...

def multibyte_split(
    source: DataChunkSource,
    delimiter: str,
    options: ParseOptions | None = None,
) -> Column: ...
def make_source(data: str) -> DataChunkSource: ...
def make_source_from_file(filename: str) -> DataChunkSource: ...
def make_source_from_bgzip_file(
    filename: str, virtual_begin: int = -1, virtual_end: int = -1
) -> DataChunkSource: ...
