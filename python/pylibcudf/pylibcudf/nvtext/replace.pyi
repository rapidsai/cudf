# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def replace_tokens(
    input: Column,
    targets: Column,
    replacements: Column,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def filter_tokens(
    input: Column,
    min_token_length: int,
    replacement: Scalar | None = None,
    delimiter: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
