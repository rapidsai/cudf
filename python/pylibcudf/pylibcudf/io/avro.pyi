# Copyright (c) 2024, NVIDIA CORPORATION.
from rmm.pylibrmm.stream import Stream

from pylibcudf.io.types import SourceInfo, TableWithMetadata

__all__ = ["AvroReaderOptions", "AvroReaderOptionsBuilder", "read_avro"]

class AvroReaderOptions:
    @staticmethod
    def builder(source: SourceInfo) -> AvroReaderOptionsBuilder: ...

class AvroReaderOptionsBuilder:
    def columns(col_names: list[str]) -> AvroReaderOptionsBuilder: ...
    def skip_rows(skip_rows: int) -> AvroReaderOptionsBuilder: ...
    def num_rows(num_rows: int) -> AvroReaderOptionsBuilder: ...
    def build(self) -> AvroReaderOptions: ...

def read_avro(
    options: AvroReaderOptions, stream: Stream = None
) -> TableWithMetadata: ...
