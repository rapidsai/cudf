# Copyright (c) 2024, NVIDIA CORPORATION.

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, overload

import pyarrow as pa

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar
from pylibcudf.table import Table
from pylibcudf.types import DataType

@dataclass
class ColumnMetadata:
    name: str = ...
    children_meta: list[ColumnMetadata] = ...

@overload
def from_arrow(obj: pa.DataType) -> DataType: ...
@overload
def from_arrow(
    obj: pa.Scalar[Any], *, data_type: DataType | None = None
) -> Scalar: ...
@overload
def from_arrow(obj: pa.Array[Any]) -> Column: ...
@overload
def from_arrow(obj: pa.Table) -> Table: ...
@overload
def to_arrow(
    obj: DataType,
    *,
    precision: int | None = None,
    fields: Iterable[pa.Field[pa.DataType] | tuple[str, pa.DataType]]
    | Mapping[str, pa.DataType]
    | None = None,
    value_type: pa.DataType | None = None,
) -> pa.DataType: ...
@overload
def to_arrow(
    obj: Table, metadata: list[ColumnMetadata | str] | None = None
) -> pa.Table: ...
@overload
def to_arrow(
    obj: Column, metadata: ColumnMetadata | str | None = None
) -> pa.Array[Any]: ...
@overload
def to_arrow(
    obj: Scalar, metadata: ColumnMetadata | str | None = None
) -> pa.Scalar[Any]: ...
def from_dlpack(managed_tensor: Any) -> Table: ...
def to_dlpack(input: Table) -> Any: ...
