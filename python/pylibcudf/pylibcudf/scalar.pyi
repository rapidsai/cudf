# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from rmm.pylibrmm.stream import Stream

from pylibcudf.column import Column
from pylibcudf.types import DataType

NpGeneric = type[Any]

PaScalar = type[Any]

class Scalar:
    def __init__(self): ...
    def type(self) -> DataType: ...
    def is_valid(self, stream: Stream) -> bool: ...
    @staticmethod
    def empty_like(column: Column, stream: Stream | None = None) -> Scalar: ...
    def to_arrow(
        self, metadata: list | str | None = None, stream: Stream | None = None
    ) -> PaScalar: ...
    @staticmethod
    def from_arrow(
        pa_val: Any,
        dtype: DataType | None = None,
        stream: Stream | None = None,
    ) -> Scalar: ...
    @classmethod
    def from_py(
        cls,
        py_val: Any,
        dtype: DataType | None = None,
        stream: Stream | None = None,
    ) -> Scalar: ...
    @classmethod
    def from_numpy(
        cls, np_val: NpGeneric, stream: Stream | None = None
    ) -> Scalar: ...
    def to_py(
        self, stream: Stream | None = None
    ) -> None | int | float | str | bool: ...
