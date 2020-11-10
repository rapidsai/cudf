from typing import Tuple, Union, TypeVar, Optional

from cudf._typing import DtypeObj, Dtype, ScalarObj
from cudf.core.buffer import Buffer
from cudf.core.column import ColumnBase


T = TypeVar("T")

class Column:
    _data: Union[Buffer, None]
    _mask: Buffer
    _base_data: Buffer
    _base_mask: Buffer
    _dtype: DtypeObj
    _offset: int
    _null_count: int
    _children: Tuple["ColumnBase", ...]
    _base_children: Tuple["ColumnBase", ...]

    def __init__(
        self,
        data: Union[Buffer, None],
        dtype: Dtype,
        size: int = None,
        mask: Buffer = None,
        offset: int = None,
        null_count: int = None,
        children: Tuple["ColumnBase", ...] = (),
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

    def set_base_mask(self, value: Union[Buffer, None]) -> None:
        ...

    def set_mask(self: T, value: Union[Buffer, None]) -> T:
        ...

    @property
    def null_count(self) -> int:
        ...

    @property
    def offset(self) -> int:
        ...

    @property
    def base_children(self) -> Tuple["ColumnBase", ...]:
        ...

    @property
    def children(self) -> Tuple["ColumnBase", ...]:
        ...

    def set_base_children(self, value: Tuple["ColumnBase", ...]) -> None:
        ...

    def _mimic_inplace(self, other_col: "ColumnBase", inplace=False) -> Optional["ColumnBase"]:
        ...

    @staticmethod
    def from_scalar(
        val: ScalarObj,
        size: int
    ) -> "ColumnBase":  # TODO: This should be Scalar, not ScalarObj
        ...
