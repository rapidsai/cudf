# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def find_multiple(
    input: Column,
    targets: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def contains_multiple(
    input: Column,
    targets: Column,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
