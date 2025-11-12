# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.table import Table

def concatenate[ColumnOrTable: (Column, Table)](
    objects: list[ColumnOrTable],
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> ColumnOrTable: ...
