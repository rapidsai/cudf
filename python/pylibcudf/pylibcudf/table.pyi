# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf._interop_helpers import ArrowLike
from pylibcudf.column import Column
from pylibcudf.types import DataType

class Table:
    def __init__(self, column: list[Column]): ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def shape(self) -> tuple[int, int]: ...
    def columns(self) -> list[Column]: ...
    def to_arrow(self, metadata: list) -> ArrowLike: ...
    # Private methods below are included because polars is currently using them,
    # but we want to remove stubs for these private methods eventually
    def _to_schema(self, metadata: Any = None) -> Any: ...
    def _to_host_array(self, stream: Stream) -> Any: ...
    @staticmethod
    def from_arrow(
        arrow_like: ArrowLike,
        dtype: DataType | None = None,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Table: ...
