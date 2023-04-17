# Copyright (c) 2023, NVIDIA CORPORATION.

import os
import zoneinfo
from functools import lru_cache
from typing import Optional, Tuple

import cudf
from cudf._lib.labeling import label_bins
from cudf._lib.search import search_sorted
from cudf._lib.timezone import make_timezone_transition_table
from cudf.core.column.column import build_column
from cudf.core.column.datetime import DatetimeColumn, DatetimeTZColumn
from cudf.core.dataframe import DataFrame


@lru_cache(maxsize=20)
def get_tz_data(zone_name):
    """
    Return timezone data (transition times and UTC offsets) for the
    given IANA time zone.

    Parameters
    ----------
    zone_name: str
        IANA time zone name

    Returns
    -------
    DataFrame with two columns containing the transition times ("dt")
    and corresponding UTC offsets ("offset").
    """
    try:
        # like zoneinfo, we first look in TZPATH
        return _find_and_read_tzfile_tzpath(zone_name)
    except zoneinfo.ZoneInfoNotFoundError:
        # if that fails, we fall back to using `tzdata`
        return _find_and_read_tzfile_tzdata(zone_name)


def _find_and_read_tzfile_tzpath(zone_name):
    for search_path in zoneinfo.TZPATH:
        if os.path.isfile(os.path.join(search_path, zone_name)):
            return _read_tzfile_as_frame(search_path, zone_name)
    raise zoneinfo.ZoneInfoNotFoundError(zone_name)


def _find_and_read_tzfile_tzdata(zone_name):
    import importlib.resources

    package_base = "tzdata.zoneinfo"
    try:
        return _read_tzfile_as_frame(
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


def _read_tzfile_as_frame(tzdir, zone_name):
    dt, offsets = make_timezone_transition_table(tzdir, zone_name)
    return DataFrame._from_columns([dt, offsets], ["dt", "offsets"])


def _find_ambiguous_and_nonexistent(
    data: DatetimeColumn, zone_name: str
) -> Optional[Tuple]:
    """
    Recognize ambiguous and nonexistent timestamps for the given timezone.
    """
    tz_zone = get_tz_data(zone_name)
    dt = tz_zone["dt"]
    offsets = tz_zone["offsets"].astype(f"timedelta64[{data._time_unit}]")

    if len(offsets) == 1:  # no transitions
        return False, False

    dt, offsets, old_offsets = (
        dt[1:]._column,
        offsets[1:]._column,
        offsets[:-1]._column,
    )

    # Assume we have two clocks:
    # - Clock 1 is turned forward or backwards correctly at transition times
    # - Clock 2 makes no changes at transition times
    clock_1 = dt + offsets
    clock_2 = dt + old_offsets

    # At the start of the ambiguous time period, Clock 1 (which has
    # been turned back) reads less than Clock 2:
    cond = clock_1 < clock_2
    ambiguous_begin = clock_1.apply_boolean_mask(cond)

    # The end of the ambiguous time period is what Clock 2 reads at
    # transition time:
    ambiguous_end = clock_2.apply_boolean_mask(cond)
    ambiguous = label_bins(
        data, ambiguous_begin, True, ambiguous_end, False
    ).notnull()

    # At the start of the non-existent time period, Clock 2 reads less
    # than Clock 1 (which has been turned forward):
    cond = clock_1 > clock_2
    nonexistent_begin = clock_2.apply_boolean_mask(cond)

    # The end of the non-existent time period is what Clock 1 reads
    # at transition time:
    nonexistent_end = clock_1.apply_boolean_mask(cond)
    nonexistent = label_bins(
        data, nonexistent_begin, True, nonexistent_end, False
    ).notnull()

    return ambiguous, nonexistent


def localize(data, zone_name):
    dtype = cudf.DatetimeTZDtype(data._time_unit, zone_name)
    ambiguous, nonexistent = _find_ambiguous_and_nonexistent(data, zone_name)
    localized = data._scatter_by_column(
        data.isnull() and (ambiguous or nonexistent),
        cudf.Scalar(cudf.NA, dtype=data.dtype),
    )
    gmt_data = local_to_utc(localized, zone_name)
    return build_column(
        data=gmt_data.data,
        dtype=dtype,
        mask=localized.mask,
        size=gmt_data.size,
        offset=gmt_data.offset,
    )


def utc_to_local(data: DatetimeTZColumn, zone: str) -> DatetimeColumn:
    tz_zone = get_tz_data(zone)
    time_starts = tz_zone._data["dt"].astype(data.dtype.base)
    indices = search_sorted([time_starts], [data], "right")
    gmt_offsets = (
        tz_zone["offsets"]
        ._column.astype(f"timedelta64[{data._time_unit}]")
        .take(indices, nullify=True)
    )
    # apply each offset to get the time in `zone`:
    return data + gmt_offsets


def local_to_utc(data, zone):
    tz_zone = get_tz_data(zone)
    time_starts = (
        tz_zone._data["dt"] + tz_zone._data["offsets"].astype("timedelta64[s]")
    ).astype(data.dtype.base)
    indices = search_sorted([time_starts], [data], "right")
    gmt_offsets = (
        tz_zone["offsets"]
        ._column.astype(f"timedelta64[{data._time_unit}]")
        .take(indices, nullify=True)
    )
    # apply each offset to get the time in `zone`:
    return data - gmt_offsets
