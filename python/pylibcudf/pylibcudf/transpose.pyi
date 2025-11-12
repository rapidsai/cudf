# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf.table import Table

def transpose(
    input_table: Table,
    stream: Stream | None = None,
    mr: DeviceMemoryResource | None = None,
) -> Table: ...
