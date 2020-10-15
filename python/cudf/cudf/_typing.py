from typing import TYPE_CHECKING, TypeVar, Union

import numpy as np

if TYPE_CHECKING:
    from pandas import Interval, Period, Timedelta, Timestamp
    from pandas.api.extensions import ExtensionDtype

# Many of these are from
# https://github.com/pandas-dev/pandas/blob/master/pandas/_typing.py

Dtype = Union["ExtensionDtype", str, np.dtype]
DtypeObj = Union["ExtensionDtype", np.dtype]

# scalars
PythonScalar = Union[str, int, float, bool]
DatetimeLikeScalar = TypeVar(
    "DatetimeLikeScalar", "Period", "Timestamp", "Timedelta"
)
PandasScalar = Union["Period", "Timestamp", "Timedelta", "Interval"]
ScalarObj = Union[PythonScalar, PandasScalar]
