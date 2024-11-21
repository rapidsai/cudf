# Copyright (c) 2021, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Literal

from typing_extensions import Self

import pylibcudf as plc

from cudf._typing import Dtype, DtypeObj, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase

class Column:
    _data: Buffer | None
    _mask: Buffer | None
    _base_data: Buffer | None
    _base_mask: Buffer | None
    _dtype: DtypeObj
    _size: int
    _offset: int
    _null_count: int
    _children: tuple[ColumnBase, ...]
    _base_children: tuple[ColumnBase, ...]
    _distinct_count: dict[bool, int]

    def __init__(
        self,
        data: Buffer | None,
        size: int,
        dtype: Dtype,
        mask: Buffer | None = None,
        offset: int | None = None,
        null_count: int | None = None,
        children: tuple[ColumnBase, ...] = (),
    ) -> None: ...
    @property
    def base_size(self) -> int: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def size(self) -> int: ...
    @property
    def base_data(self) -> Buffer | None: ...
    @property
    def data(self) -> Buffer | None: ...
    @property
    def data_ptr(self) -> int: ...
    def set_base_data(self, value: Buffer) -> None: ...
    @property
    def nullable(self) -> bool: ...
    def has_nulls(self, include_nan: bool = False) -> bool: ...
    @property
    def base_mask(self) -> Buffer | None: ...
    @property
    def mask(self) -> Buffer | None: ...
    @property
    def mask_ptr(self) -> int: ...
    def set_base_mask(self, value: Buffer | None) -> None: ...
    def set_mask(self, value: ColumnBase | Buffer | None) -> Self: ...
    @property
    def null_count(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def base_children(self) -> tuple[ColumnBase, ...]: ...
    @property
    def children(self) -> tuple[ColumnBase, ...]: ...
    def set_base_children(self, value: tuple[ColumnBase, ...]) -> None: ...
    def _mimic_inplace(
        self, other_col: ColumnBase, inplace=False
    ) -> Self | None: ...

    # TODO: The val parameter should be Scalar, not ScalarLike
    @staticmethod
    def from_scalar(val: ScalarLike, size: int) -> ColumnBase: ...
    @staticmethod
    def from_pylibcudf(
        col: plc.Column, data_ptr_exposed: bool = False
    ) -> ColumnBase: ...
    def to_pylibcudf(self, mode: Literal["read", "write"]) -> plc.Column: ...
