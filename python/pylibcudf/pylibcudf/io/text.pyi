# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.io.text import DataChunkSource, ParseOptions

def make_source(data: str) -> DataChunkSource: ...
def make_source_from_file(filename: str) -> DataChunkSource: ...
def make_source_from_bgzip_file(
    filename: str,
    virtual_begin: int = -1,
    virtual_end: int = -1,
) -> DataChunkSource: ...
def multibyte_split(
    source: DataChunkSource,
    delimiter: str,
    options: ParseOptions | None = None,
    stream: Stream | None = None,
) -> Column: ...
