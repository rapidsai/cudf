# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def url_encode(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def url_decode(
    input: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
