# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from cudf.core.scalar import Scalar


class Timestamp:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pd_ts_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        ts = pd.Timestamp(*args, **pd_ts_kwargs)
        self._scalar = Scalar(ts, dtype=kwargs.get("dtype"))

    @property
    def value(self) -> int:
        return pd.Timestamp(self._scalar.value).value

    @property
    def year(self) -> int:
        return pd.Timestamp(self._scalar.value).year

    @property
    def month(self) -> int:
        return pd.Timestamp(self._scalar.value).month

    @property
    def day(self) -> int:
        return pd.Timestamp(self._scalar.value).month

    @property
    def hour(self) -> int:
        return pd.Timestamp(self._scalar.value).hour

    @property
    def minute(self) -> int:
        return pd.Timestamp(self._scalar.value).minute

    @property
    def second(self) -> int:
        return pd.Timestamp(self._scalar.value).second

    @property
    def microsecond(self) -> int:
        return pd.Timestamp(self._scalar.value).microsecond

    @property
    def nanosecond(self) -> int:
        return pd.Timestamp(self._scalar.value).nanosecond

    def __repr__(self):
        return pd.Timestamp(self._scalar._host_value).__repr__()

    @property
    def asm8(self) -> np.datetime64:
        return self._scalar.value

    def to_pandas(self):
        return pd.Timestamp(self._scalar.value)

    @classmethod
    def from_pandas(cls, obj: pd.Timestamp):
        return cls(obj)
