# Copyright (c) 2024, NVIDIA CORPORATION.

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

def repeat_strings(
    input: Column,
    repeat_times: Column | int,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
