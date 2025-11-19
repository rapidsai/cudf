# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, TypedDict

from rmm.pylibrmm.device_buffer import DeviceBuffer
from rmm.pylibrmm.memory_resource import DeviceMemoryResource
from rmm.pylibrmm.stream import Stream

from pylibcudf._interop_helpers import ArrowLike
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType

class ArrayInterfaceBase(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    data: None | tuple[int, bool]
    version: int
    strides: None | tuple[int, ...]
    descr: None | list[tuple[Any, ...]]

class ArrayInterface(ArrayInterfaceBase):
    mask: None | "SupportsArrayInterface"

class CudaArrayInterface(ArrayInterfaceBase):
    stream: None | int
    mask: None | "SupportsCudaArrayInterface"

class SupportsCudaArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> CudaArrayInterface: ...

class SupportsArrayInterface(Protocol):
    @property
    def __array_interface__(self) -> ArrayInterface: ...

class Column:
    def __init__(
        self,
        data_type: DataType,
        size: int,
        data: gpumemoryview | None,
        mask: gpumemoryview | None,
        null_count: int,
        offset: int,
        children: list[Column],
    ) -> None: ...
    def type(self) -> DataType: ...
    def child(self, index: int) -> Column: ...
    def size(self) -> int: ...
    def null_count(self) -> int: ...
    def offset(self) -> int: ...
    def data(self) -> gpumemoryview | None: ...
    def null_mask(self) -> gpumemoryview | None: ...
    def children(self) -> list[Column]: ...
    def num_children(self) -> int: ...
    def copy(
        self,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def device_buffer_size(self) -> int: ...
    def with_mask(
        self, mask: gpumemoryview | None, null_count: int
    ) -> Column: ...
    def list_view(self) -> ListColumnView: ...
    @staticmethod
    def from_scalar(
        scalar: Scalar,
        size: int,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    def to_scalar(
        self,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Scalar: ...
    @staticmethod
    def all_null_like(
        like: Column,
        size: int,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    @staticmethod
    def from_rmm_buffer(
        buff: DeviceBuffer, dtype: DataType, size: int, children: list[Column]
    ) -> Column: ...
    def to_arrow(
        self, metadata: list | str | None = None, stream: Stream | None = None
    ) -> ArrowLike: ...
    # Private methods below are included because polars is currently using them,
    # but we want to remove stubs for these private methods eventually
    def _to_schema(self, metadata: Any = None) -> Any: ...
    def _to_host_array(self, stream: Stream) -> Any: ...
    @staticmethod
    def from_arrow(
        obj: ArrowLike,
        dtype: DataType | None = None,
        stream: Stream | None = None,
        mr: DeviceMemoryResource | None = None,
    ) -> Column: ...
    @classmethod
    def from_cuda_array_interface(
        cls, obj: SupportsCudaArrayInterface, stream: Stream | None = None
    ) -> Column: ...
    @classmethod
    def from_array_interface(
        cls, obj: SupportsArrayInterface, stream: Stream | None = None
    ) -> Column: ...
    @staticmethod
    def from_array(
        cls,
        obj: SupportsCudaArrayInterface | SupportsArrayInterface,
        stream: Stream | None = None,
    ) -> Column: ...
    @staticmethod
    def struct_from_children(children: Sequence[Column]) -> Column: ...
    @staticmethod
    def from_iterable_of_py(
        obj: Iterable,
        dtype: DataType | None = None,
        stream: Stream | None = None,
    ) -> Column: ...

class ListColumnView:
    def __init__(self, column: Column): ...
    def child(self) -> Column: ...
    def offsets(self) -> Column: ...

def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int] | None, itemsize: int
) -> bool: ...
