# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class BPEMergePairs:
    def __init__(
        self,
        merge_pairs: Column,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ): ...

def byte_pair_encoding(
    input: Column,
    merge_pairs: BPEMergePairs,
    separator: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
