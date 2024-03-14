# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, TypeVar, Union

import numpy as np
from pandas import Period, Timedelta, Timestamp
from pandas.api.extensions import ExtensionDtype

if TYPE_CHECKING:
    import cudf

# Backwards compat: mypy >= 0.790 rejects Type[NotImplemented], but
# NotImplementedType is only introduced in 3.10
if sys.version_info >= (3, 10):
    from types import NotImplementedType
else:
    NotImplementedType = Any

# Many of these are from
# https://github.com/pandas-dev/pandas/blob/master/pandas/_typing.py

Dtype = Union["ExtensionDtype", str, np.dtype]
DtypeObj = Union["ExtensionDtype", np.dtype]

# scalars
DatetimeLikeScalar = TypeVar(
    "DatetimeLikeScalar", Period, Timestamp, Timedelta
)
ScalarLike = Any

# columns
ColumnLike = Any

# binary operation
ColumnBinaryOperand = Union["cudf.Scalar", "cudf.core.column.ColumnBase"]

DataFrameOrSeries = Union["cudf.Series", "cudf.DataFrame"]
SeriesOrIndex = Union["cudf.Series", "cudf.core.index.BaseIndex"]
SeriesOrSingleColumnIndex = Union["cudf.Series", "cudf.core.index.Index"]

# Groupby aggregation
AggType = Union[str, Callable]
MultiColumnAggType = Union[
    AggType, Iterable[AggType], Dict[Any, Iterable[AggType]]
]
