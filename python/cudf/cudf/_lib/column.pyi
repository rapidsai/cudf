# Copyright (c) 2021, NVIDIA CORPORATION.

from __future__ import annotations
from typing import Tuple, Union, TypeVar, Optional

from cudf._typing import DtypeObj, Dtype, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase


T = TypeVar("T")

class Column:
    _data: Optional[Buffer]
    _mask: Optional[Buffer]
    _base_data: Optional[Buffer]
    _base_mask: Optional[Buffer]
    _dtype: DtypeObj
    _size: int
    _offset: int
    _null_count: int
    _children: Tuple[ColumnBase, ...]
    _base_children: Tuple[ColumnBase, ...]

    def __init__(
        self,
        data: Optional[Buffer],
        size: int,
        dtype: Dtype,
        mask: Optional[Buffer] = None,
        offset: int = None,
        null_count: int = None,
        children: Tuple[ColumnBase, ...] = (),
    ) -> None:
        ...

    @property
    def base_size(self) -> int:
        ...

    @property
    def dtype(self) -> DtypeObj:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def base_data(self) -> Optional[Buffer]:
        ...

    @property
    def base_data_ptr(self) -> int:
        ...

    @property
    def data(self) -> Optional[Buffer]:
        ...

    @property
    def data_ptr(self) -> int:
        ...

    def set_base_data(self, value: Buffer) -> None:
        ...

    @property
    def nullable(self) -> bool:
        ...

    @property
    def has_nulls(self) -> bool:
        ...

    @property
    def base_mask(self) -> Optional[Buffer]:
        ...

    @property
    def base_mask_ptr(self) -> int:
        ...

    @property
    def mask(self) -> Optional[Buffer]:
        ...

    @property
    def mask_ptr(self) -> int:
        ...

    def set_base_mask(self, value: Optional[Buffer]) -> None:
        ...

    def set_mask(self: T, value: Optional[Buffer]) -> T:
        ...

    @property
    def null_count(self) -> int:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def base_children(self) -> Tuple[ColumnBase, ...]:
        ...

    @property
    def children(self) -> Tuple[ColumnBase, ...]:
        ...

    def set_base_children(self, value: Tuple[ColumnBase, ...]) -> None:
        ...

    def _mimic_inplace(self, other_col: ColumnBase, inplace=False) -> Optional[ColumnBase]:
        ...

    @staticmethod
    def from_scalar(
        val: ScalarLike,
        size: int
    ) -> ColumnBase:  # TODO: This should be Scalar, not ScalarLike
        ...
