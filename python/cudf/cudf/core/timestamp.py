# Copyright (c) 2024, NVIDIA CORPORATION.

from __future__ import annotations

from datetime import datetime, timedelta

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
    def value(self) -> np.datetime64:
        return super().value

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

    @classmethod
    def from_scalar(cls, obj: Scalar):
        return cls(obj.value)

    def _to_scalar(self):
        return Scalar(self.value)

    def __add__(self, other: timedelta | np.timedelta64):
        return self.from_scalar(self._to_scalar() + other)

    def __radd__(self, other: timedelta):
        return self + other

    def __sub__(
        self, other: datetime | timedelta | np.timedelta64
    ) -> pd.Timedelta:
        if isinstance(other, datetime):
            return pd.Timedelta(self.value - other)
        elif isinstance(other, self.__class__):
            return pd.Timedelta(self.value - other.value)
        elif isinstance(other, (timedelta, np.timedelta64)):
            return self.from_scalar(self._to_scalar() - other)
        else:
            raise TypeError(
                f"Subtraction not supported between types {type(self)} and {type(other)}"
            )

    @property
    def as_unit(self):
        raise NotImplementedError(
            "The attribute 'as_unit' is not implemented."
        )

    @property
    def day_of_week(self):
        raise NotImplementedError(
            "The attribute 'day_of_week' is not implemented."
        )

    @property
    def day_of_year(self):
        raise NotImplementedError(
            "The attribute 'day_of_year' is not implemented."
        )

    @property
    def dayofweek(self):
        raise NotImplementedError(
            "The attribute 'dayofweek' is not implemented."
        )

    @property
    def dayofyear(self):
        raise NotImplementedError(
            "The attribute 'dayofyear' is not implemented."
        )

    @property
    def days_in_month(self):
        raise NotImplementedError(
            "The attribute 'days_in_month' is not implemented."
        )

    @property
    def daysinmonth(self):
        raise NotImplementedError(
            "The attribute 'daysinmonth' is not implemented."
        )

    @property
    def fold(self):
        raise NotImplementedError("The attribute 'fold' is not implemented.")

    @property
    def is_leap_year(self):
        raise NotImplementedError(
            "The attribute 'is_leap_year' is not implemented."
        )

    @property
    def is_month_end(self):
        raise NotImplementedError(
            "The attribute 'is_month_end' is not implemented."
        )

    @property
    def is_month_start(self):
        raise NotImplementedError(
            "The attribute 'is_month_start' is not implemented."
        )

    @property
    def is_quarter_end(self):
        raise NotImplementedError(
            "The attribute 'is_quarter_end' is not implemented."
        )

    @property
    def is_quarter_start(self):
        raise NotImplementedError(
            "The attribute 'is_quarter_start' is not implemented."
        )

    @property
    def is_year_end(self):
        raise NotImplementedError(
            "The attribute 'is_year_end' is not implemented."
        )

    @property
    def is_year_start(self):
        raise NotImplementedError(
            "The attribute 'is_year_start' is not implemented."
        )

    @property
    def min(self):
        raise NotImplementedError("The attribute 'min' is not implemented.")

    @property
    def quarter(self):
        raise NotImplementedError(
            "The attribute 'quarter' is not implemented."
        )

    @property
    def resolution(self):
        raise NotImplementedError(
            "The attribute 'resolution' is not implemented."
        )

    @property
    def timestamp(self):
        raise NotImplementedError(
            "The attribute 'timestamp' is not implemented."
        )

    @property
    def tz(self):
        raise NotImplementedError("The attribute 'tz' is not implemented.")

    @property
    def tzinfo(self):
        raise NotImplementedError("The attribute 'tzinfo' is not implemented.")

    @property
    def unit(self):
        raise NotImplementedError("The attribute 'unit' is not implemented.")

    @property
    def week(self):
        raise NotImplementedError("The attribute 'week' is not implemented.")

    @property
    def weekday(self):
        raise NotImplementedError(
            "The attribute 'weekday' is not implemented."
        )

    @property
    def weekofyear(self):
        raise NotImplementedError(
            "The attribute 'weekofyear' is not implemented."
        )

    def astimezone(self, tz=None):
        raise NotImplementedError(
            "The method 'astimezone' is not implemented."
        )

    def ceil(self, freq=None):
        raise NotImplementedError("The method 'ceil' is not implemented.")

    @classmethod
    def combine(cls, date, time):
        raise NotImplementedError("The method 'combine' is not implemented.")

    def ctime(self):
        raise NotImplementedError("The method 'ctime' is not implemented.")

    def date(self):
        raise NotImplementedError("The method 'date' is not implemented.")

    def day_name(self):
        raise NotImplementedError("The method 'day_name' is not implemented.")

    def dst(self):
        raise NotImplementedError("The method 'dst' is not implemented.")

    def floor(self, freq=None):
        raise NotImplementedError("The method 'floor' is not implemented.")

    @classmethod
    def fromisoformat(cls, date_string):
        raise NotImplementedError(
            "The method 'fromisoformat' is not implemented."
        )

    @classmethod
    def fromordinal(cls, n):
        raise NotImplementedError(
            "The method 'fromordinal' is not implemented."
        )

    @classmethod
    def fromtimestamp(cls, timestamp, tz=None):
        raise NotImplementedError(
            "The method 'fromtimestamp' is not implemented."
        )

    def isocalendar(self):
        raise NotImplementedError(
            "The method 'isocalendar' is not implemented."
        )

    def isoformat(self, sep="T"):
        raise NotImplementedError("The method 'isoformat' is not implemented.")

    def isoweekday(self):
        raise NotImplementedError(
            "The method 'isoweekday' is not implemented."
        )

    def max(self):
        raise NotImplementedError("The method 'max' is not implemented.")

    def month_name(self):
        raise NotImplementedError(
            "The method 'month_name' is not implemented."
        )

    def normalize(self):
        raise NotImplementedError("The method 'normalize' is not implemented.")

    @classmethod
    def now(cls, tz=None):
        raise NotImplementedError("The method 'now' is not implemented.")

    def replace(self, **kwargs):
        raise NotImplementedError("The method 'replace' is not implemented.")

    def round(self, freq=None):
        raise NotImplementedError("The method 'round' is not implemented.")

    def strftime(self, format):
        raise NotImplementedError("The method 'strftime' is not implemented.")

    @classmethod
    def strptime(cls, date_string, format):
        raise NotImplementedError("The method 'strptime' is not implemented.")

    def time(self):
        raise NotImplementedError("The method 'time' is not implemented.")

    def timetuple(self):
        raise NotImplementedError("The method 'timetuple' is not implemented.")

    def timetz(self):
        raise NotImplementedError("The method 'timetz' is not implemented.")

    def to_datetime64(self):
        raise NotImplementedError(
            "The method 'to_datetime64' is not implemented."
        )

    def to_julian_date(self):
        raise NotImplementedError(
            "The method 'to_julian_date' is not implemented."
        )

    def to_numpy(self):
        raise NotImplementedError("The method 'to_numpy' is not implemented.")

    def to_period(self, freq=None):
        raise NotImplementedError("The method 'to_period' is not implemented.")

    def to_pydatetime(self):
        raise NotImplementedError(
            "The method 'to_pydatetime' is not implemented."
        )

    def today(self):
        raise NotImplementedError("The method 'today' is not implemented.")

    def toordinal(self):
        raise NotImplementedError("The method 'toordinal' is not implemented.")

    def tz_convert(self, tz):
        raise NotImplementedError(
            "The method 'tz_convert' is not implemented."
        )

    def tz_localize(self, tz):
        raise NotImplementedError(
            "The method 'tz_localize' is not implemented."
        )

    def tzname(self):
        raise NotImplementedError("The method 'tzname' is not implemented.")

    @classmethod
    def utcfromtimestamp(cls, timestamp):
        raise NotImplementedError(
            "The method 'utcfromtimestamp' is not implemented."
        )

    @classmethod
    def utcnow(cls):
        raise NotImplementedError("The method 'utcnow' is not implemented.")

    def utcoffset(self):
        raise NotImplementedError("The method 'utcoffset' is not implemented.")

    def utctimetuple(self):
        raise NotImplementedError(
            "The method 'utctimetuple' is not implemented."
        )
