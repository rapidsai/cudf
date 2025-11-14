# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm import DeviceBuffer
from rmm.mr import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.table import Table

class PackedColumns:
    def __init__(self): ...
    def release(
        self, stream: Stream | None = None
    ) -> tuple[memoryview[bytes], gpumemoryview]: ...

def pack(input: Table, stream: Stream | None = None) -> PackedColumns: ...
def unpack(input: PackedColumns, stream: Stream | None = None) -> Table: ...
def unpack_from_memoryviews(
    metadata: memoryview[bytes],
    gpu_data: gpumemoryview,
    stream: Stream | None = None,
) -> Table: ...

class ChunkedPack:
    def __init__(self): ...
    @staticmethod
    def create(
        input: Table,
        user_buffer_size: int,
        stream: Stream | None = None,
        temp_mr: DeviceMemoryResource | None = None,
    ) -> ChunkedPack: ...
    def has_next(self) -> bool: ...
    def get_total_contiguous_size(self) -> int: ...
    def next(self, buf: DeviceBuffer) -> int: ...
    def build_metadata(self) -> memoryview[bytes]: ...
    def pack_to_host(
        self, buf: DeviceBuffer
    ) -> tuple[memoryview[bytes], memoryview[bytes]]: ...
