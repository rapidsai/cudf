# Copyright (c) 2021, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf._typing import DataFrameOrSeries
from cudf.core.groupby.groupby import (
    DataFrameGroupBy,
    GroupBy,
    Grouper,
    SeriesGroupBy,
    _Grouping,
)


# TODO: is this the best inheritance strategy to use?
class _Resampler(GroupBy):

    grouping: "_ResampleGrouping"

    def __init__(self, obj, by, axis=None, kind=None):
        if not isinstance(by, Grouper) and by.freq:
            # TODO
            raise ValueError()

        by = _ResampleGrouping(obj, by)
        super().__init__(obj, by=by)

    def agg(self, func):
        result = super().agg(func)
        index = cudf.Index(
            self.grouping.bin_labels, name=self.grouping.names[0]
        )
        return result._align_to_index(index, how="right")

    def asfreq(self):
        return self.obj._align_to_index(self.grouping.bin_labels, how="right")

    def _scan_fill(self, method: str, limit: int) -> DataFrameOrSeries:
        # TODO: can this be more efficient?

        # first, compute the outer join between `self.obj` and the `bin_labels`
        # to get the sampling "gaps":
        upsampled = self.obj._align_to_index(
            self.grouping.bin_labels, how="outer"
        )

        # fill the gaps:
        filled = upsampled.fillna(method=method)

        # filter the result to only include the values corresponding
        # to the bin labels:
        return filled._align_to_index(self.grouping.bin_labels, how="right")


class DataFrameResampler(_Resampler, DataFrameGroupBy):
    pass


class SeriesResampler(_Resampler, SeriesGroupBy):
    pass


class _ResampleGrouping(_Grouping):

    bin_labels: cudf.Index

    def _handle_frequency_grouper(self, by):
        import cudf._lib.labeling

        freq = by.freq
        key = by.key
        label = by.label
        closed = by.closed

        if key is None:
            # then assume that the key is the index of `self._obj`:
            key_column = self._obj.index._column

        else:
            key_column = self._obj._data[key]

        # conver freq to a pd.DateOffset:
        if isinstance(freq, str):
            offset = pd.tseries.frequencies.to_offset(freq)
        elif isinstance(freq, cudf.DateOffset):
            offset = freq.to_pandas()
        else:
            if not isinstance(freq, pd.DateOffset):
                raise TypeError(
                    "Unsupported type for freq: {type(freq).__name__}"
                )
            offset = freq

        if offset.freqstr == "M" or offset.freqstr.startswith("W-"):
            label = "right" if label is None else label
            closed = "right" if closed is None else closed
        else:
            label = "left" if label is None else label
            closed = "left" if closed is None else closed

        start, end = _get_timestamp_range_edges(
            pd.Timestamp(key_column.min()),
            pd.Timestamp(key_column.max()),
            offset,
            closed=closed,
        )

        end += offset
        bin_labels = cudf.date_range(start=start, end=end, freq=freq,)
        bins = cudf._lib.labeling.label_bins(
            key_column,
            left_edges=bin_labels[:-1]._column,
            left_inclusive=(closed == "left"),
            right_edges=bin_labels[1:]._column,
            right_inclusive=(closed == "right"),
        )

        if label == "right":
            bin_labels = bin_labels[1:]
        else:
            bin_labels = bin_labels[:-1]

        nbins = bins.max() + 1
        if len(bin_labels) > nbins:
            # if we have more labels than bins:
            bin_labels = bin_labels[:-1]

        key_name = key if key is not None else self._obj.index.name
        bin_labels.name = key_name

        if key is not None:
            self._named_columns.append(key_name)
        self.names.append(key_name)
        self.bin_labels = bin_labels
        self._key_columns.append(bin_labels._column.take(bins))


# NOTE: this function is vendored from Pandas
def _get_timestamp_range_edges(
    first, last, freq, closed="left", origin="start_day", offset=None
):
    """
    Adjust the `first` Timestamp to the preceding Timestamp that resides on
    the provided offset. Adjust the `last` Timestamp to the following
    Timestamp that resides on the provided offset. Input Timestamps that
    already reside on the offset will be adjusted depending on the type of
    offset and the `closed` parameter.

    Parameters
    ----------
    first : pd.Timestamp
        The beginning Timestamp of the range to be adjusted.
    last : pd.Timestamp
        The ending Timestamp of the range to be adjusted.
    freq : pd.DateOffset
        The dateoffset to which the Timestamps will be adjusted.
    closed : {'right', 'left'}, default None
        Which side of bin interval is closed.
    origin : {'epoch', 'start', 'start_day'} or Timestamp, default 'start_day'
        The timestamp on which to adjust the grouping. The timezone of origin
        must match the timezone of the index.  If a timestamp is not used,
        these values are also supported:

        - 'epoch': `origin` is 1970-01-01
        - 'start': `origin` is the first value of the timeseries
        - 'start_day': `origin` is the first day at midnight of the timeseries
    offset : pd.Timedelta, default is None
        An offset timedelta added to the origin.

    Returns
    -------
    A tuple of length 2, containing the adjusted pd.Timestamp objects.
    """
    from pandas.tseries.offsets import Day, Tick

    if isinstance(freq, Tick):
        index_tz = first.tz
        if isinstance(origin, pd.Timestamp) and (origin.tz is None) != (
            index_tz is None
        ):
            raise ValueError(
                "The origin must have the same timezone as the index."
            )
        elif origin == "epoch":
            # set the epoch based on the timezone to have similar bins results
            # when resampling on the same kind of indexes on different
            # timezones
            origin = pd.Timestamp("1970-01-01", tz=index_tz)

        if isinstance(freq, Day):
            # _adjust_dates_anchored assumes 'D' means 24H, but first/last
            # might contain a DST transition (23H, 24H, or 25H).
            # So "pretend" the dates are naive when adjusting the endpoints
            first = first.tz_localize(None)
            last = last.tz_localize(None)
            if isinstance(origin, pd.Timestamp):
                origin = origin.tz_localize(None)

        first, last = _adjust_dates_anchored(
            first, last, freq, closed=closed, origin=origin, offset=offset
        )
        if isinstance(freq, Day):
            first = first.tz_localize(index_tz)
            last = last.tz_localize(index_tz)
    else:
        first = first.normalize()
        last = last.normalize()

        if closed == "left":
            first = pd.Timestamp(freq.rollback(first))
        else:
            first = pd.Timestamp(first - freq)

        last = pd.Timestamp(last + freq)

    return first, last


# NOTE: this function is vendored from Pandas
def _adjust_dates_anchored(
    first, last, freq, closed="right", origin="start_day", offset=None
):
    # First and last offsets should be calculated from the start day to fix an
    # error cause by resampling across multiple days when a one day period is
    # not a multiple of the frequency. See GH 8683
    # To handle frequencies that are not multiple or divisible by a day we let
    # the possibility to define a fixed origin timestamp. See GH 31809
    origin_nanos = 0  # origin == "epoch"
    if origin == "start_day":
        origin_nanos = first.normalize().value
    elif origin == "start":
        origin_nanos = first.value
    elif isinstance(origin, pd.Timestamp):
        origin_nanos = origin.value
    origin_nanos += offset.value if offset else 0

    # GH 10117 & GH 19375. If first and last contain timezone information,
    # Perform the calculation in UTC in order to avoid localizing on an
    # Ambiguous or Nonexistent time.
    first_tzinfo = first.tzinfo
    last_tzinfo = last.tzinfo
    if first_tzinfo is not None:
        first = first.tz_convert("UTC")
    if last_tzinfo is not None:
        last = last.tz_convert("UTC")

    foffset = (first.value - origin_nanos) % freq.nanos
    loffset = (last.value - origin_nanos) % freq.nanos

    if closed == "right":
        if foffset > 0:
            # roll back
            fresult = first.value - foffset
        else:
            fresult = first.value - freq.nanos

        if loffset > 0:
            # roll forward
            lresult = last.value + (freq.nanos - loffset)
        else:
            # already the end of the road
            lresult = last.value
    else:  # closed == 'left'
        if foffset > 0:
            fresult = first.value - foffset
        else:
            # start of the road
            fresult = first.value

        if loffset > 0:
            # roll forward
            lresult = last.value + (freq.nanos - loffset)
        else:
            lresult = last.value + freq.nanos
    fresult = pd.Timestamp(fresult)
    lresult = pd.Timestamp(lresult)
    if first_tzinfo is not None:
        fresult = fresult.tz_localize("UTC").tz_convert(first_tzinfo)
    if last_tzinfo is not None:
        lresult = lresult.tz_localize("UTC").tz_convert(last_tzinfo)
    return fresult, lresult
