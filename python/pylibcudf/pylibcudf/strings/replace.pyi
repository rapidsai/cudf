# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

def replace(
    input: Column,
    target: Scalar,
    repl: Scalar,
    maxrepl: int = -1,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_multiple(
    input: Column,
    target: Column,
    repl: Column,
    maxrepl: int = -1,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
def replace_slice(
    input: Column,
    repl: Scalar | None = None,
    start: int = 0,
    stop: int = -1,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Column: ...
