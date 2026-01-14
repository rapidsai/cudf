# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
import pandas as pd
from packaging.version import parse

if TYPE_CHECKING:
    from pandas.api.extensions import ExtensionDtype

    import cudf

# Many of these are from
# https://github.com/pandas-dev/pandas/blob/master/pandas/_typing.py

# Dtype should ideally only used for public facing APIs
Dtype = Union["ExtensionDtype", str, np.dtype]
# DtypeObj should be used otherwise
DtypeObj = Union["ExtensionDtype", np.dtype]

# scalars
DatetimeLikeScalar = TypeVar(
    "DatetimeLikeScalar", pd.Period, pd.Timestamp, pd.Timedelta
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

if parse(pd.__version__) >= parse("3.0.0"):
    from pandas.api.typing import NoDefault
else:
    NoDefault = Any
