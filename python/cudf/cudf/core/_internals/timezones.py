# Copyright (c) 2023, NVIDIA CORPORATION.

import os
import zoneinfo
from functools import lru_cache
from typing import cast

import cudf
from cudf._lib.labeling import label_bins
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


def localize(data: DatetimeColumn, tz: str) -> DatetimeTZColumn:
    """
    Recognize ambiguous or nonexistent timestamps and set them to NaT
    """
    tz_zone = get_tz_data(tz)
    dtype = cudf.DatetimeTZDtype(data._time_unit, tz)
    time_start = tz_zone["dt"]
    gmt_offset = (
        tz_zone["offsets"]
        .astype("timedelta64[s]")
        .astype(f"timedelta64[{data._time_unit}]")
    )

    local_time_new_offsets = time_start[1:]._column + gmt_offset[1:]._column
    local_time_old_offsets = time_start[1:]._column + gmt_offset[:-1]._column

    if len(local_time_old_offsets) == 0:  # no transitions
        return cast(
            DatetimeTZColumn,
            build_column(
                data=data.base_data,
                dtype=dtype,
                size=data.size,
                mask=data.mask,
                offset=data.offset,
            ),
        )

    # ambiguous time periods happen when the clock is
    # moved backward after the transition
    ambiguous_begin = local_time_new_offsets.apply_boolean_mask(
        local_time_new_offsets < local_time_old_offsets
    )
    ambiguous_end = local_time_old_offsets.apply_boolean_mask(
        local_time_new_offsets < local_time_old_offsets
    )

    # nonexistent time periods happen when the clock is
    # moved forward after the transition
    nonexistent_begin = local_time_old_offsets.apply_boolean_mask(
        local_time_new_offsets > local_time_old_offsets
    )
    nonexistent_end = local_time_new_offsets.apply_boolean_mask(
        local_time_new_offsets > local_time_old_offsets
    )

    ambiguous = label_bins(
        data, ambiguous_begin, True, ambiguous_end, False
    ).notnull()
    nonexistent = label_bins(
        data, nonexistent_begin, True, nonexistent_end, False
    ).notnull()

    set_to_nat = ambiguous._binaryop(nonexistent, "__or__")
    localized_data = data._scatter_by_column(
        set_to_nat, cudf.Scalar(None, dtype=data.dtype)
    )

    return cast(
        DatetimeTZColumn,
        build_column(
            data=data.base_data,
            dtype=dtype,
            size=data.size,
            mask=localized_data.mask,
            offset=localized_data.offset,
        ),
    )
