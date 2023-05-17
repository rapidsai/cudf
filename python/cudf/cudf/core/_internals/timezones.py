# Copyright (c) 2023, NVIDIA CORPORATION.

import os
import zoneinfo
from functools import lru_cache
from typing import Tuple, cast

import numpy as np
import pandas as pd

import cudf
from cudf._lib.labeling import label_bins
from cudf._lib.search import search_sorted
from cudf._lib.timezone import make_timezone_transition_table
from cudf.core.column.column import as_column, build_column
from cudf.core.column.datetime import DatetimeColumn, DatetimeTZColumn
from cudf.core.dataframe import DataFrame
from cudf.utils.dtypes import _get_base_dtype


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
    DataFrame with two columns containing the transition times
    ("transition_times") and corresponding UTC offsets ("offsets").
    """
    try:
        # like zoneinfo, we first look in TZPATH
        tz_table = _find_and_read_tzfile_tzpath(zone_name)
    except zoneinfo.ZoneInfoNotFoundError:
        # if that fails, we fall back to using `tzdata`
        tz_table = _find_and_read_tzfile_tzdata(zone_name)
    return tz_table


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
    transition_times_and_offsets = make_timezone_transition_table(
        tzdir, zone_name
    )

    if not transition_times_and_offsets:
        # this happens for UTC-like zones
        min_date = np.int64(np.iinfo("int64").min + 1).astype("M8[s]")
        transition_times_and_offsets = as_column([min_date]), as_column(
            [np.timedelta64(0, "s")]
        )

    return DataFrame._from_columns(
        transition_times_and_offsets, ["transition_times", "offsets"]
    )


def _find_ambiguous_and_nonexistent(
    data: DatetimeColumn, zone_name: str
) -> Tuple:
    """
    Recognize ambiguous and nonexistent timestamps for the given timezone.

    Returns a tuple of columns, both of "bool" dtype and of the same
    size as `data`, that respectively indicate ambiguous and
    nonexistent timestamps in `data` with the value `True`.

    Ambiguous and/or nonexistent timestamps are only possible if any
    transitions occur in the time zone database for the given timezone.
    If no transitions occur, the tuple `(False, False)` is returned.
    """
    tz_data_for_zone = get_tz_data(zone_name)
    transition_times = tz_data_for_zone["transition_times"]
    offsets = tz_data_for_zone["offsets"].astype(
        f"timedelta64[{data._time_unit}]"
    )

    if len(offsets) == 1:  # no transitions
        return False, False

    transition_times, offsets, old_offsets = (
        transition_times[1:]._column,
        offsets[1:]._column,
        offsets[:-1]._column,
    )

    # Assume we have two clocks at the moment of transition:
    # - Clock 1 is turned forward or backwards correctly
    # - Clock 2 makes no changes
    clock_1 = transition_times + offsets
    clock_2 = transition_times + old_offsets

    # At the start of an ambiguous time period, Clock 1 (which has
    # been turned back) reads less than Clock 2:
    cond = clock_1 < clock_2
    ambiguous_begin = clock_1.apply_boolean_mask(cond)

    # The end of an ambiguous time period is what Clock 2 reads at
    # the moment of transition:
    ambiguous_end = clock_2.apply_boolean_mask(cond)
    ambiguous = label_bins(
        data,
        left_edges=ambiguous_begin,
        left_inclusive=True,
        right_edges=ambiguous_end,
        right_inclusive=False,
    ).notnull()

    # At the start of a non-existent time period, Clock 2 reads less
    # than Clock 1 (which has been turned forward):
    cond = clock_1 > clock_2
    nonexistent_begin = clock_2.apply_boolean_mask(cond)

    # The end of the non-existent time period is what Clock 1 reads
    # at the moment of transition:
    nonexistent_end = clock_1.apply_boolean_mask(cond)
    nonexistent = label_bins(
        data,
        left_edges=nonexistent_begin,
        left_inclusive=True,
        right_edges=nonexistent_end,
        right_inclusive=False,
    ).notnull()

    return ambiguous, nonexistent


def localize(
    data: DatetimeColumn, zone_name: str, ambiguous, nonexistent
) -> DatetimeTZColumn:
    if ambiguous != "NaT":
        raise NotImplementedError(
            "Only ambiguous='NaT' is currently supported"
        )
    if nonexistent != "NaT":
        raise NotImplementedError(
            "Only nonexistent='NaT' is currently supported"
        )
    if isinstance(data, DatetimeTZColumn):
        raise ValueError(
            "Already localized. "
            "Use `tz_convert` to convert between time zones."
        )
    dtype = pd.DatetimeTZDtype(data._time_unit, zone_name)
    ambiguous, nonexistent = _find_ambiguous_and_nonexistent(data, zone_name)
    localized = cast(
        DatetimeColumn,
        data._scatter_by_column(
            data.isnull() | (ambiguous | nonexistent),
            cudf.Scalar(cudf.NA, dtype=data.dtype),
        ),
    )
    gmt_data = local_to_utc(localized, zone_name)
    return cast(
        DatetimeTZColumn,
        build_column(
            data=gmt_data.base_data,
            dtype=dtype,
            mask=localized.base_mask,
            size=gmt_data.size,
            offset=gmt_data.offset,
        ),
    )


def convert(data: DatetimeTZColumn, zone_name: str) -> DatetimeTZColumn:
    if not isinstance(data, DatetimeTZColumn):
        raise TypeError(
            "Cannot convert from timezone-naive timestamps to "
            "timezone-aware timestamps. For that, "
            "use `tz_localize`."
        )
    if zone_name == str(data.dtype.tz):
        return data.copy()
    utc_time = data._utc_time
    out = cast(
        DatetimeTZColumn,
        build_column(
            data=utc_time.base_data,
            dtype=pd.DatetimeTZDtype(data._time_unit, zone_name),
            mask=utc_time.base_mask,
            size=utc_time.size,
            offset=utc_time.offset,
        ),
    )
    return out


def utc_to_local(data: DatetimeColumn, zone_name: str) -> DatetimeColumn:
    tz_data_for_zone = get_tz_data(zone_name)
    transition_times, offsets = tz_data_for_zone._columns
    transition_times = transition_times.astype(_get_base_dtype(data.dtype))
    indices = search_sorted([transition_times], [data], "right") - 1
    offsets_from_utc = offsets.take(indices, nullify=True)
    return data + offsets_from_utc


def local_to_utc(data: DatetimeColumn, zone_name: str) -> DatetimeColumn:
    tz_data_for_zone = get_tz_data(zone_name)
    transition_times, offsets = tz_data_for_zone._columns
    transition_times_local = (transition_times + offsets).astype(data.dtype)
    indices = search_sorted([transition_times_local], [data], "right") - 1
    offsets_to_utc = offsets.take(indices, nullify=True)
    return data - offsets_to_utc
