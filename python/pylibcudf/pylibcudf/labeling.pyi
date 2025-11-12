# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column

class Inclusive(IntEnum):
    YES = ...
    NO = ...

def label_bins(
    input: Column,
    left_edges: Column,
    left_inclusive: Inclusive,
    right_edges: Column,
    right_inclusive: Inclusive,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
