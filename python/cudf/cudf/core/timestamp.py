# Copyright (c) 2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from cudf.core.scalar import Scalar


class Timestamp(Scalar):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        pd_ts_kwargs = {k: v for k, v in kwargs.items() if k != "dtype"}
        ts = pd.Timestamp(*args, **pd_ts_kwargs)
        super().__init__(ts)

    @property
    def value(self) -> int:
        return pd.Timestamp(super().value).value

    @property
    def year(self) -> int:
        return pd.Timestamp(super().value).year

    @property
    def month(self) -> int:
        return pd.Timestamp(super().value).month

    @property
    def day(self) -> int:
        return pd.Timestamp(super().value).day

    @property
    def hour(self) -> int:
        return pd.Timestamp(super().value).hour

    @property
    def minute(self) -> int:
        return pd.Timestamp(super().value).minute

    @property
    def second(self) -> int:
        return pd.Timestamp(super().value).second

    @property
    def microsecond(self) -> int:
        return pd.Timestamp(super().value).microsecond

    @property
    def nanosecond(self) -> int:
        return pd.Timestamp(super().value).nanosecond

    def __repr__(self):
        return pd.Timestamp(self.value).__repr__()

    @property
    def asm8(self) -> np.datetime64:
        return super().value

    def to_pandas(self):
        return pd.Timestamp(super().value)

    @classmethod
    def from_pandas(cls, obj: pd.Timestamp):
        return cls(obj)
