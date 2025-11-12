# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def slice_strings(
    input: Column,
    start: Column | Scalar | None = None,
    stop: Column | Scalar | None = None,
    step: Scalar | None = None,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
