from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from pandas import Period, Timedelta, Timestamp
    from pandas.api.extensions import ExtensionDtype

    import cudf

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
BinaryOperand = Union["cudf.Scalar", "cudf.core.column.ColumnBase"]
