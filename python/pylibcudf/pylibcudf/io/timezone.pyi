# Copyright (c) 2024-2025, NVIDIA CORPORATION.

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.table import Table

def make_timezone_transition_table(
    tzif_dir: str,
    timezone_name: str,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
