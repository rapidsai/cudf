from typing import Tuple, Union

from cudf._typing import DtypeObj, Dtype
from cudf.core.buffer import Buffer


class Column:
    _data: Buffer
    _mask: Buffer
    _base_data: Buffer
    _base_mask: Buffer
    _dtype: DtypeObj
    _offset: int
    _null_count: int
    _children: Tuple["Column", ...]
    _base_children: Tuple["Column", ...]

    def __init__(
        self,
        data: Buffer,
        dtype: Dtype,
        size: int = None,
        mask: Buffer = None,
        offset: int = None,
        null_count: int = None,
        children: Tuple["Column", ...] = (),
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
    def base_data(self) -> Buffer:
        ...

    @property
    def base_data_ptr(self) -> int:
        ...

    @property
    def data(self) -> Buffer:
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
    def base_mask(self) -> Buffer:
        ...

    @property
    def base_mask_ptr(self) -> int:
        ...

    @property
    def mask(self) -> Buffer:
        ...

    @property
    def mask_ptr(self) -> int:
        ...

    def set_base_mask(self, value: Buffer) -> None:
        ...

    def set_mask(self, value: Buffer) -> None:
        ...

    @property
    def null_count(self) -> int:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def base_children(self) -> Tuple["Column", ...]:
        ...

    @property
    def children(self) -> Tuple["Column", ...]:
        ...

    def set_base_children(self, value: Tuple["Column", ...]) -> None:
        ...

    def _mimic_inplace(self, other_col, inplace=False):
        ...
