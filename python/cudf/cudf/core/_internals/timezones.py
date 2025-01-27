# Copyright (c) 2023-2024, NVIDIA CORPORATION.
from __future__ import annotations

import datetime
import os
import zoneinfo
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

import pylibcudf as plc

import cudf
from cudf._lib.column import Column

if TYPE_CHECKING:
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.timedelta import TimeDeltaColumn


def get_compatible_timezone(dtype: pd.DatetimeTZDtype) -> pd.DatetimeTZDtype:
    """Convert dtype.tz object to zoneinfo object if possible."""
    tz = dtype.tz
    if isinstance(tz, zoneinfo.ZoneInfo):
        return dtype
    if cudf.get_option("mode.pandas_compatible"):
        raise NotImplementedError(
            f"{tz} must be a zoneinfo.ZoneInfo object in pandas_compatible mode."
        )
    elif (tzname := getattr(tz, "zone", None)) is not None:
        # pytz-like
        key = tzname
    elif (tz_file := getattr(tz, "_filename", None)) is not None:
        # dateutil-like
        key = tz_file.split("zoneinfo/")[-1]
    elif isinstance(tz, datetime.tzinfo):
        # Try to get UTC-like tzinfos
        reference = datetime.datetime.now()
        key = tz.tzname(reference)
        if not (isinstance(key, str) and key.lower() == "utc"):
            raise NotImplementedError(f"cudf does not support {tz}")
    else:
        raise NotImplementedError(f"cudf does not support {tz}")
    new_tz = zoneinfo.ZoneInfo(key)
    return pd.DatetimeTZDtype(dtype.unit, new_tz)


@lru_cache(maxsize=20)
def get_tz_data(zone_name: str) -> tuple[DatetimeColumn, TimeDeltaColumn]:
    """
    Return timezone data (transition times and UTC offsets) for the
    given IANA time zone.

    Parameters
    ----------
    zone_name: str
        IANA time zone name

    Returns
    -------
    Tuple with two columns containing the transition times
    and corresponding UTC offsets.
    """
    try:
        # like zoneinfo, we first look in TZPATH
        tz_table = _find_and_read_tzfile_tzpath(zone_name)
    except zoneinfo.ZoneInfoNotFoundError:
        # if that fails, we fall back to using `tzdata`
        tz_table = _find_and_read_tzfile_tzdata(zone_name)
    return tz_table


def _find_and_read_tzfile_tzpath(
    zone_name: str,
) -> tuple[DatetimeColumn, TimeDeltaColumn]:
    for search_path in zoneinfo.TZPATH:
        if os.path.isfile(os.path.join(search_path, zone_name)):
            return _read_tzfile_as_columns(search_path, zone_name)
    raise zoneinfo.ZoneInfoNotFoundError(zone_name)


def _find_and_read_tzfile_tzdata(
    zone_name: str,
) -> tuple[DatetimeColumn, TimeDeltaColumn]:
    import importlib.resources

    package_base = "tzdata.zoneinfo"
    try:
        return _read_tzfile_as_columns(
            str(importlib.resources.files(package_base)), zone_name
        )
    # TODO: make it so that the call to libcudf raises a
    # FileNotFoundError instead of a RuntimeError
    except (ImportError, FileNotFoundError, UnicodeEncodeError, RuntimeError):
        # the "except" part of this try-except is basically vendored
        # from the zoneinfo library.
        #
        # There are three types of exception that can be raised that all amount
        # to "we cannot find this key":
        #
        # ImportError: If package_name doesn't exist (e.g. if tzdata is not
        #   installed, or if there's an error in the folder name like
        #   Amrica/New_York)
        # FileNotFoundError: If resource_name doesn't exist in the package
        #   (e.g. Europe/Krasnoy)
        # UnicodeEncodeError: If package_name or resource_name are not UTF-8,
        #   such as keys containing a surrogate character.
        raise zoneinfo.ZoneInfoNotFoundError(zone_name)


def _read_tzfile_as_columns(
    tzdir: str, zone_name: str
) -> tuple[DatetimeColumn, TimeDeltaColumn]:
    plc_table = plc.io.timezone.make_timezone_transition_table(
        tzdir, zone_name
    )
    transition_times_and_offsets = [
        Column.from_pylibcudf(col) for col in plc_table.columns()
    ]

    if not transition_times_and_offsets:
        from cudf.core.column.column import as_column

        # this happens for UTC-like zones
        min_date = np.int64(np.iinfo("int64").min + 1).astype("M8[s]")
        return (as_column([min_date]), as_column([np.timedelta64(0, "s")]))  # type: ignore[return-value]
    return tuple(transition_times_and_offsets)  # type: ignore[return-value]


def check_ambiguous_and_nonexistent(
    ambiguous: Literal["NaT"], nonexistent: Literal["NaT"]
) -> tuple[Literal["NaT"], Literal["NaT"]]:
    if ambiguous != "NaT":
        raise NotImplementedError(
            "Only ambiguous='NaT' is currently supported"
        )
    if nonexistent != "NaT":
        raise NotImplementedError(
            "Only nonexistent='NaT' is currently supported"
        )
    return ambiguous, nonexistent
