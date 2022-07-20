import os

import pandas as pd

import cudf
from cudf._lib.labeling import label_bins
from cudf._lib.search import search_sorted

TZDIR = os.path.join(os.path.dirname(__file__), "TimeZoneDB.csv")

countries = cudf.read_csv(os.path.join(TZDIR, "country.csv"))
tz = cudf.read_csv(
    os.path.join(TZDIR, "time_zone.csv"),
    names="zone_name,country_code,abbreviation,time_start,gmt_offset,dst".split(
        ","
    ),
    dtype=["str", "str", "str", "int64", int, int],
)

tz["time_start"] = tz["time_start"].astype("datetime64[s]")
tz["gmt_offset"] = tz["gmt_offset"]


def get_tz_for_zone(zone):
    return tz[tz.zone_name == zone].reset_index(drop=True)


def time_start_bins(data, tz):
    # given a Series of timestamps `data`,
    # return the indexes to the closest `time_start`
    # corresponding to each element.
    time_starts_in_zone = tz._data["time_start"] + tz._data[
        "gmt_offset"
    ].astype("timedelta64[s]")
    bin_indices = search_sorted(
        time_starts_in_zone,
        data._column.astype("datetime64[s]"),
    ).fillna(-1)
    return tz.index.take(bin_indices)


def tz_localize(data, zone):
    # recognize ambiguous or nonexistent timestamps and set them to NaT
    tz_zone = get_tz_for_zone(zone)

    time_start = tz_zone["time_start"]
    gmt_offset = (
        tz_zone["gmt_offset"]
        .astype("timedelta64[s]")
        .astype(f"timedelta64[{data._time_unit}]")
    )

    local_time_new_offsets = time_start[1:]._column + gmt_offset[1:]._column
    local_time_old_offsets = time_start[1:]._column + gmt_offset[:-1]._column

    if len(local_time_old_offsets) == 0:  # no transitions
        return

    # ambiguous time periods happen when the clock is moved backward after the transition
    ambiguous_begin = local_time_new_offsets.apply_boolean_mask(
        local_time_new_offsets < local_time_old_offsets
    )
    ambiguous_end = local_time_old_offsets.apply_boolean_mask(
        local_time_new_offsets < local_time_old_offsets
    )

    # nonexistent time periods happen when the clock is moved forward after the transition
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

    return ambiguous._binaryop(nonexistent, "__or__")


def from_gmt(data, zone):
    tz_zone = tz[tz._data["zone_name"] == zone]
    time_starts = tz_zone._data["time_start"].astype(data.dtype.base)
    indices = search_sorted([time_starts], [data], "left")
    gmt_offsets = (
        tz_zone["gmt_offset"]
        ._column.astype("timedelta64[s]")
        .astype(f"timedelta64[{data._time_unit}]")
        .take(indices, nullify=True)
    )
    # apply each offset to get the time in `zone`:
    return data + gmt_offsets


def to_gmt(data, zone):
    tz_zone = tz[tz._data["zone_name"] == zone]
    time_starts = (
        tz_zone._data["time_start"]
        + tz_zone._data["gmt_offset"].astype("timedelta64[s]")
    ).astype(data.dtype.base)
    indices = search_sorted([time_starts], [data], "left")
    gmt_offsets = (
        tz_zone["gmt_offset"]
        ._column.astype("timedelta64[s]")
        .astype(f"timedelta64[{data._time_unit}]")
        .take(indices, nullify=True)
    )
    # apply each offset to get the time in `zone`:
    return data - gmt_offsets


def tz_convert(data, from_timezone, to_timezone):
    return from_gmt(to_gmt(data, from_timezone), to_timezone)


def tz_add(data, offset, zone):
    return from_gmt(to_gmt(data, zone) + offset, zone)
