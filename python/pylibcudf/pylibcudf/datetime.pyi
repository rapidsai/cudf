# Copyright (c) 2024, NVIDIA CORPORATION.

from enum import IntEnum, auto

from pylibcudf.column import Column
from pylibcudf.scalar import Scalar

class DatetimeComponent(IntEnum):
    YEAR = auto()
    MONTH = auto()
    DAY = auto()
    WEEKDAY = auto()
    HOUR = auto()
    MINUTE = auto()
    SECOND = auto()
    MILLISECOND = auto()
    MICROSECOND = auto()
    NANOSECOND = auto()

class RoundingFrequency(IntEnum):
    DAY = auto()
    HOUR = auto()
    MINUTE = auto()
    SECOND = auto()
    MILLISECOND = auto()
    MICROSECOND = auto()
    NANOSECOND = auto()

def extract_millisecond_fraction(input: Column) -> Column: ...
def extract_microsecond_fraction(input: Column) -> Column: ...
def extract_nanosecond_fraction(input: Column) -> Column: ...
def extract_datetime_component(
    input: Column, component: DatetimeComponent
) -> Column: ...
def ceil_datetimes(input: Column, freq: RoundingFrequency) -> Column: ...
def floor_datetimes(input: Column, freq: RoundingFrequency) -> Column: ...
def round_datetimes(input: Column, freq: RoundingFrequency) -> Column: ...
def add_calendrical_months(
    timestamps: Column, months: Column | Scalar
) -> Column: ...
def day_of_year(input: Column) -> Column: ...
def is_leap_year(input: Column) -> Column: ...
def last_day_of_month(input: Column) -> Column: ...
def extract_quarter(input: Column) -> Column: ...
def days_in_month(input: Column) -> Column: ...
