# Copyright (c) 2021, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Dict, Optional, Tuple

from typing_extensions import Self

from cudf._typing import Dtype, DtypeObj, ScalarLike
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase

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
    _distinct_count: Dict[bool, int]

    def __init__(
        self,
        data: Optional[Buffer],
        size: int,
        dtype: Dtype,
        mask: Optional[Buffer] = None,
        offset: Optional[int] = None,
        null_count: Optional[int] = None,
        children: Tuple[ColumnBase, ...] = (),
    ) -> None: ...
    @property
    def base_size(self) -> int: ...
    @property
    def dtype(self) -> DtypeObj: ...
    @property
    def size(self) -> int: ...
    @property
    def base_data(self) -> Optional[Buffer]: ...
    @property
    def data(self) -> Optional[Buffer]: ...
    @property
    def data_ptr(self) -> int: ...
    def set_base_data(self, value: Buffer) -> None: ...
    @property
    def nullable(self) -> bool: ...
    def has_nulls(self, include_nan: bool = False) -> bool: ...
    @property
    def base_mask(self) -> Optional[Buffer]: ...
    @property
    def mask(self) -> Optional[Buffer]: ...
    @property
    def mask_ptr(self) -> int: ...
    def set_base_mask(self, value: Optional[Buffer]) -> None: ...
    def set_mask(self, value: Optional[Buffer]) -> Self: ...
    @property
    def null_count(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def base_children(self) -> Tuple[ColumnBase, ...]: ...
    @property
    def children(self) -> Tuple[ColumnBase, ...]: ...
    def set_base_children(self, value: Tuple[ColumnBase, ...]) -> None: ...
    def _mimic_inplace(
        self, other_col: ColumnBase, inplace=False
    ) -> Optional[Self]: ...

    # TODO: The val parameter should be Scalar, not ScalarLike
    @staticmethod
    def from_scalar(val: ScalarLike, size: int) -> ColumnBase: ...
