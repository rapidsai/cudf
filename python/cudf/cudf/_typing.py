# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
from pandas import Period, Timedelta, Timestamp

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionDtype

    import cudf

# Backwards compat: mypy >= 0.790 rejects Type[NotImplemented], but
# NotImplementedType is only introduced in 3.10
if sys.version_info >= (3, 10):
    from types import NotImplementedType
else:
    NotImplementedType = Any

# Many of these are from
# https://github.com/pandas-dev/pandas/blob/master/pandas/_typing.py

# Dtype should ideally only used for public facing APIs
Dtype = Union["ExtensionDtype", str, np.dtype]
# DtypeObj should be used otherwise
DtypeObj = Union["ExtensionDtype", np.dtype]

# scalars
DatetimeLikeScalar = TypeVar(
    "DatetimeLikeScalar", Period, Timestamp, Timedelta
)
ScalarLike = Any

# columns
ColumnLike = Any

# binary operation
ColumnBinaryOperand = Union[ScalarLike, "cudf.core.column.ColumnBase"]

DataFrameOrSeries = Union["cudf.Series", "cudf.DataFrame"]
SeriesOrIndex = Union["cudf.Series", "cudf.Index"]
SeriesOrSingleColumnIndex = Union["cudf.Series", "cudf.Index"]

# Groupby aggregation
AggType = Union[str, Callable]  # noqa: UP007
MultiColumnAggType = Union[  # noqa: UP007
    AggType, Iterable[AggType], dict[Any, Iterable[AggType]]
]

Axis = Literal[0, 1, "index", "columns"]
