# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from rmm.pylibrmm.device_buffer import DeviceBuffer

from pylibcudf._interop_helpers import ArrowLike
from pylibcudf.gpumemoryview import gpumemoryview
from pylibcudf.scalar import Scalar
from pylibcudf.types import DataType

class ArrayInterface(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    data: None | tuple[int, bool]
    version: int
    strides: None | tuple[int, ...]
    descr: None | list[tuple[Any, ...]]
    mask: None | "SupportsArrayInterface"

class CudaArrayInterface(ArrayInterface):
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
    def copy(self) -> Column: ...
    def device_buffer_size(self) -> int: ...
    def with_mask(
        self, mask: gpumemoryview | None, null_count: int
    ) -> Column: ...
    def list_view(self) -> ListColumnView: ...
    @staticmethod
    def from_scalar(scalar: Scalar, size: int) -> Column: ...
    def to_scalar(self) -> Column: ...
    @staticmethod
    def all_null_like(like: Column, size: int) -> Column: ...
    @staticmethod
    def from_rmm_buffer(
        buff: DeviceBuffer, dtype: DataType, size: int, children: list[Column]
    ) -> Column: ...
    @staticmethod
    def from_arrow(
        obj: ArrowLike, dtype: DataType | None = None
    ) -> Column: ...
    @classmethod
    def from_cuda_array_interface(
        cls, obj: SupportsCudaArrayInterface
    ) -> Column: ...
    @classmethod
    def from_array_interface(cls, obj: SupportsArrayInterface) -> Column: ...
    @staticmethod
    def from_array(
        cls, obj: SupportsCudaArrayInterface | SupportsArrayInterface
    ) -> Column: ...

class ListColumnView:
    def __init__(self, column: Column): ...
    def child(self) -> Column: ...
    def offsets(self) -> Column: ...

def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int], itemsize: int
) -> bool: ...
