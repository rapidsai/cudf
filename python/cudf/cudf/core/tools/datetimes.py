# Copyright (c) 2019-2024, NVIDIA CORPORATION.

import math
import re
import warnings
from typing import Literal, Optional, Sequence, Union

import cupy as cp
import numpy as np
import pandas as pd
import pandas.tseries.offsets as pd_offset
from typing_extensions import Self

import cudf
from cudf import _lib as libcudf
from cudf._lib.strings.convert.convert_integers import (
    is_integer as cpp_is_integer,
)
from cudf.api.types import is_integer, is_scalar
from cudf.core import column
from cudf.core.index import as_index

# https://github.com/pandas-dev/pandas/blob/2.2.x/pandas/core/tools/datetimes.py#L1112
_unit_map = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "hour": "h",
    "hours": "h",
    "minute": "m",
    "minutes": "m",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}

_unit_dtype_map = {
    "ns": "datetime64[ns]",
    "us": "datetime64[us]",
    "ms": "datetime64[ms]",
    "m": "datetime64[s]",
    "h": "datetime64[s]",
    "s": "datetime64[s]",
    "D": "datetime64[s]",
}


def to_datetime(
    arg,
    errors: Literal["raise", "coerce", "warn", "ignore"] = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: Optional[str] = None,
    exact: bool = True,
    unit: str = "ns",
    infer_datetime_format: bool = True,
    origin="unix",
    cache: bool = True,
):
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array,
        Series DataFrame/dict-like
        The object to convert to a datetime.
    errors : {'ignore', 'raise', 'coerce', 'warn'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'warn' : prints last exceptions as warnings and
            return the input.
        - If 'ignore', then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.
        If True, parses dates with the day first, eg 10/11/12 is parsed as
        2012-11-10.
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug, based on dateutil behavior).
    utc : bool, default False
        Whether the result should be have a UTC timezone.
    format : str, default None
        The strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse
        all the way up to nanoseconds.
        See strftime documentation for more information on choices:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the
        origin(unix epoch start).
        Example, with unit='ms' and origin='unix' (the default), this
        would calculate the number of milliseconds to the unix epoch start.
    infer_datetime_format : bool, default True
        If True and no `format` is given, attempt to infer the format of the
        datetime strings, and if it can be inferred, switch to a faster
        method of parsing them. In some cases this can increase the parsing
        speed by ~5-10x.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input:
        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp

    Examples
    --------
    Assembling a datetime from multiple columns of a DataFrame. The keys can be
    common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> import cudf
    >>> df = cudf.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> cudf.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[ns]
    >>> cudf.to_datetime(1490195805, unit='s')
    numpy.datetime64('2017-03-22T15:16:45.000000000')
    >>> cudf.to_datetime(1490195805433502912, unit='ns')
    numpy.datetime64('1780-11-20T01:02:30.494253056')
    """
    if errors not in {"ignore", "raise", "coerce", "warn"}:
        raise ValueError(
            f"errors parameter has to be either one of: "
            f"{['ignore', 'raise', 'coerce', 'warn']}, found: "
            f"{errors}"
        )
    elif errors in {"ignore", "coerce"} and not is_scalar(arg):
        raise NotImplementedError(
            f"{errors=} is not implemented when arg is not scalar-like"
        )

    if errors == "ignore":
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a "
            "future version. Use to_datetime without passing `errors` "
            "and catch exceptions explicitly instead",
            FutureWarning,
        )

    if infer_datetime_format in {None, False}:
        warnings.warn(
            "`infer_datetime_format` is deprecated and will "
            "be removed in a future version of cudf.",
            FutureWarning,
        )

    if arg is None:
        return None

    if exact is False:
        raise NotImplementedError("exact support is not yet implemented")

    if origin != "unix":
        raise NotImplementedError("origin support is not yet implemented")

    if yearfirst:
        raise NotImplementedError("yearfirst support is not yet implemented")

    if format is not None:
        if "%Z" in format or "%z" in format:
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        elif "%f" in format:
            format = format.replace("%f", "%9f")

    try:
        if isinstance(arg, cudf.DataFrame):
            # we require at least Ymd
            required = ["year", "month", "day"]
            req = list(set(required) - set(arg._data.names))
            if len(req):
                err_req = ",".join(req)
                raise ValueError(
                    f"to assemble mappings requires at least that "
                    f"[year, month, day] be specified: [{err_req}] "
                    f"is missing"
                )

            # replace passed column name with values in _unit_map
            got_units = {k: get_units(k) for k in arg._data.names}
            unit_rev = {v: k for k, v in got_units.items()}

            # keys we don't recognize
            excess = set(unit_rev.keys()) - set(_unit_map.values())
            if len(excess):
                err_excess = ",".join(excess)
                raise ValueError(
                    f"extra keys have been passed to the "
                    f"datetime assemblage: [{err_excess}]"
                )

            new_series = (
                arg[unit_rev["year"]].astype("str")
                + "-"
                + arg[unit_rev["month"]].astype("str").str.zfill(2)
                + "-"
                + arg[unit_rev["day"]].astype("str").str.zfill(2)
            )
            format = "%Y-%m-%d"
            col = new_series._column.as_datetime_column(
                "datetime64[s]", format=format
            )

            for u in ["h", "m", "s", "ms", "us", "ns"]:
                value = unit_rev.get(u)
                if value is not None and value in arg:
                    arg_col = arg._data[value]
                    if arg_col.dtype.kind in ("f"):
                        col = new_series._column.as_datetime_column(
                            "datetime64[ns]", format=format
                        )
                        break
                    elif arg_col.dtype.kind in ("O"):
                        if not cpp_is_integer(arg_col).all():
                            col = new_series._column.as_datetime_column(
                                "datetime64[ns]", format=format
                            )
                            break

            times_column = None
            for u in ["h", "m", "s", "ms", "us", "ns"]:
                value = unit_rev.get(u)
                if value is not None and value in arg:
                    current_col = arg._data[value]
                    # If the arg[value] is of int or
                    # float dtype we don't want to type-cast
                    if current_col.dtype.kind in ("O"):
                        try:
                            current_col = current_col.astype(dtype="int64")
                        except ValueError:
                            current_col = current_col.astype(dtype="float64")

                    factor = cudf.Scalar(
                        column.datetime._unit_to_nanoseconds_conversion[u]
                        / (
                            column.datetime._unit_to_nanoseconds_conversion[
                                "s"
                            ]
                            if np.datetime_data(col.dtype)[0] == "s"
                            else 1
                        )
                    )

                    if times_column is None:
                        times_column = current_col * factor
                    else:
                        times_column = times_column + (current_col * factor)
            if times_column is not None:
                col = (col.astype(dtype="int64") + times_column).astype(
                    dtype=col.dtype
                )
            col = _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
                utc=utc,
            )
            return cudf.Series(col, index=arg.index)
        else:
            col = _process_col(
                col=column.as_column(arg),
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
                utc=utc,
            )
            if isinstance(arg, (cudf.BaseIndex, pd.Index)):
                return as_index(col, name=arg.name)
            elif isinstance(arg, (cudf.Series, pd.Series)):
                return cudf.Series(col, index=arg.index, name=arg.name)
            elif is_scalar(arg):
                return col.element_indexing(0)
            else:
                return as_index(col)
    except Exception as e:
        if errors == "raise":
            raise e
        elif errors == "warn":
            import traceback

            tb = traceback.format_exc()
            warnings.warn(tb)
        elif errors == "ignore":
            pass
        elif errors == "coerce":
            return np.datetime64("nat", "ns" if unit is None else unit)
        return arg


def _process_col(
    col,
    unit: str,
    dayfirst: bool,
    infer_datetime_format: bool,
    format: Optional[str],
    utc: bool,
):
    if col.dtype.kind == "f":
        if unit not in (None, "ns"):
            factor = cudf.Scalar(
                column.datetime._unit_to_nanoseconds_conversion[unit]
            )
            col = col * factor

        if format is not None:
            # Converting to int because,
            # pandas actually creates a datetime column
            # out of float values and then creates an
            # int column out of it to parse against `format`.
            # Instead we directly cast to int and perform
            # parsing against `format`.
            col = (
                col.astype("int")
                .astype("str")
                .as_datetime_column(
                    dtype="datetime64[us]"
                    if "%f" in format
                    else "datetime64[s]",
                    format=format,
                )
            )
        else:
            col = col.as_datetime_column(dtype="datetime64[ns]")

    elif col.dtype.kind in "iu":
        if unit in ("D", "h", "m"):
            factor = cudf.Scalar(
                column.datetime._unit_to_nanoseconds_conversion[unit]
                / column.datetime._unit_to_nanoseconds_conversion["s"]
            )
            col = col * factor

        if format is not None:
            col = col.astype("str").as_datetime_column(
                dtype=_unit_dtype_map[unit], format=format
            )
        else:
            col = col.as_datetime_column(dtype=_unit_dtype_map[unit])

    elif col.dtype.kind == "O":
        if unit not in (None, "ns") or col.null_count == len(col):
            try:
                col = col.astype(dtype="int64")
            except ValueError:
                col = col.astype(dtype="float64")
            return _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
                utc=utc,
            )
        else:
            if format is None:
                if not infer_datetime_format and dayfirst:
                    raise NotImplementedError(
                        f"{dayfirst=} not implemented "
                        f"when {format=} and {infer_datetime_format=}."
                    )
                format = column.datetime.infer_format(
                    element=col.element_indexing(0),
                    dayfirst=dayfirst,
                )
            col = col.as_datetime_column(
                dtype=_unit_dtype_map[unit],
                format=format,
            )
    elif col.dtype.kind != "M":
        raise TypeError(
            f"dtype {col.dtype} cannot be converted to {_unit_dtype_map[unit]}"
        )
    if utc and not isinstance(col.dtype, pd.DatetimeTZDtype):
        return col.tz_localize("UTC")
    return col


def get_units(value):
    if value in _unit_map:
        return _unit_map[value]

    # m is case significant
    if value.lower() in _unit_map:
        return _unit_map[value.lower()]

    return value


class DateOffset:
    """
    An object used for binary ops where calendrical arithmetic
    is desired rather than absolute time arithmetic. Used to
    add or subtract a whole number of periods, such as several
    months or years, to a series or index of datetime dtype.
    Works similarly to pd.DateOffset, but stores the offset
    on the device (GPU).

    Parameters
    ----------
    n : int, default 1
        The number of time periods the offset represents.
    **kwds
        Temporal parameter that add to or replace the offset value.
        Parameters that **add** to the offset (like Timedelta):
        - months

    See Also
    --------
    pandas.DateOffset : The equivalent Pandas object that this
    object replicates

    Examples
    --------
    >>> from cudf import DateOffset
    >>> ts = cudf.Series([
    ...     "2000-01-01 00:00:00.012345678",
    ...     "2000-01-31 00:00:00.012345678",
    ...     "2000-02-29 00:00:00.012345678",
    ... ], dtype='datetime64[ns]')
    >>> ts + DateOffset(months=3)
    0   2000-04-01 00:00:00.012345678
    1   2000-04-30 00:00:00.012345678
    2   2000-05-29 00:00:00.012345678
    dtype: datetime64[ns]
    >>> ts - DateOffset(months=12)
    0   1999-01-01 00:00:00.012345678
    1   1999-01-31 00:00:00.012345678
    2   1999-02-28 00:00:00.012345678
    dtype: datetime64[ns]

    Notes
    -----
    Note that cuDF does not yet support DateOffset arguments
    that 'replace' units in the datetime data being operated on
    such as
        - year
        - month
        - week
        - day
        - hour
        - minute
        - second
        - microsecond
        - millisecond
        - nanosecond

    cuDF does not yet support rounding via a `normalize`
    keyword argument.
    """

    _UNITS_TO_CODES = {
        "nanoseconds": "ns",
        "microseconds": "us",
        "milliseconds": "ms",
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
        "weeks": "W",
        "months": "M",
        "years": "Y",
    }

    _CODES_TO_UNITS = {
        "N": "nanoseconds",
        "ns": "nanoseconds",
        "U": "microseconds",
        "us": "microseconds",
        "ms": "milliseconds",
        "L": "milliseconds",
        "s": "seconds",
        "S": "seconds",
        "m": "minutes",
        "min": "minutes",
        "T": "minutes",
        "h": "hours",
        "H": "hours",
        "D": "days",
        "W": "weeks",
        "M": "months",
        "Y": "years",
    }

    _TICK_OR_WEEK_TO_UNITS = {
        pd_offset.Week: "weeks",
        pd_offset.Day: "days",
        pd_offset.Hour: "hours",
        pd_offset.Minute: "minutes",
        pd_offset.Second: "seconds",
        pd_offset.Milli: "milliseconds",
        pd_offset.Micro: "microseconds",
        pd_offset.Nano: "nanoseconds",
    }

    _FREQSTR_REGEX = re.compile("([-+]?[0-9]*)([a-zA-Z]+)")

    def __init__(self, n=1, normalize=False, **kwds):
        if normalize:
            raise NotImplementedError(
                "normalize not yet supported for DateOffset"
            )

        all_possible_units = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
            "year",
            "month",
            "week",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "millisecond",
            "nanosecond",
        }

        supported_units = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        }

        unsupported_units = all_possible_units - supported_units

        invalid_kwds = set(kwds) - supported_units - unsupported_units
        if invalid_kwds:
            raise TypeError(
                f"Keyword arguments '{','.join(list(invalid_kwds))}'"
                " are not recognized"
            )

        unsupported_kwds = set(kwds) & unsupported_units
        if unsupported_kwds:
            raise NotImplementedError(
                f"Keyword arguments '{','.join(list(unsupported_kwds))}'"
                " are not yet supported."
            )

        if any(not is_integer(val) for val in kwds.values()):
            raise ValueError("Non-integer periods not supported")

        self._kwds = kwds
        kwds = self._combine_months_and_years(**kwds)
        kwds = self._combine_kwargs_to_seconds(**kwds)

        scalars = {}
        for k, v in kwds.items():
            if k in all_possible_units:
                # Months must be int16
                if k == "months":
                    # TODO: throw for out-of-bounds int16 values
                    dtype = "int16"
                else:
                    unit = self._UNITS_TO_CODES[k]
                    dtype = cudf.dtype(f"timedelta64[{unit}]")
                scalars[k] = cudf.Scalar(v, dtype=dtype)

        self._scalars = scalars

    @property
    def kwds(self):
        return self._kwds

    def _combine_months_and_years(self, **kwargs):
        # TODO: if months is zero, don't do a binop
        kwargs["months"] = kwargs.pop("years", 0) * 12 + kwargs.pop(
            "months", 0
        )
        return kwargs

    def _combine_kwargs_to_seconds(self, **kwargs):
        """
        Combine days, weeks, hours and minutes to a single
        scalar representing the total seconds
        """
        seconds = 0
        seconds += kwargs.pop("weeks", 0) * 604800
        seconds += kwargs.pop("days", 0) * 86400
        seconds += kwargs.pop("hours", 0) * 3600
        seconds += kwargs.pop("minutes", 0) * 60
        seconds += kwargs.pop("seconds", 0)

        if seconds > np.iinfo("int64").max:
            raise NotImplementedError(
                "Total days + weeks + hours + minutes + seconds can not exceed"
                f" {np.iinfo('int64').max} seconds"
            )

        if seconds != 0:
            kwargs["seconds"] = seconds
        return kwargs

    def _datetime_binop(
        self, datetime_col, op, reflect=False
    ) -> column.DatetimeColumn:
        if reflect and op == "__sub__":
            raise TypeError(
                f"Can not subtract a {type(datetime_col).__name__}"
                f" from a {type(self).__name__}"
            )
        if op not in {"__add__", "__sub__"}:
            raise TypeError(
                f"{op} not supported between {type(self).__name__}"
                f" and {type(datetime_col).__name__}"
            )
        if not self._is_no_op:
            if "months" in self._scalars:
                rhs = self._generate_months_column(len(datetime_col), op)
                datetime_col = libcudf.datetime.add_months(datetime_col, rhs)

            for unit, value in self._scalars.items():
                if unit != "months":
                    value = -value if op == "__sub__" else value
                    datetime_col += cudf.core.column.as_column(
                        value, length=len(datetime_col)
                    )

        return datetime_col

    def _generate_months_column(self, size, op):
        months = self._scalars["months"]
        months = -months if op == "__sub__" else months
        # TODO: pass a scalar instead of constructing a column
        # https://github.com/rapidsai/cudf/issues/6990
        col = cudf.core.column.as_column(months, length=size)
        return col

    @property
    def _is_no_op(self) -> bool:
        # some logic could be implemented here for more complex cases
        # such as +1 year, -12 months
        return all(i == 0 for i in self._kwds.values())

    def __neg__(self):
        new_scalars = {k: -v for k, v in self._kwds.items()}
        return DateOffset(**new_scalars)

    def __repr__(self):
        includes = []
        for unit in sorted(self._UNITS_TO_CODES):
            val = self._kwds.get(unit, None)
            if val is not None:
                includes.append(f"{unit}={val}")
        unit_data = ", ".join(includes)
        repr_str = f"<{self.__class__.__name__}: {unit_data}>"

        return repr_str

    @classmethod
    def _from_freqstr(cls, freqstr: str) -> Self:
        """
        Parse a string and return a DateOffset object
        expects strings of the form 3D, 25W, 10ms, 42ns, etc.
        """
        match = cls._FREQSTR_REGEX.match(freqstr)

        if match is None:
            raise ValueError(f"Invalid frequency string: {freqstr}")

        numeric_part = match.group(1)
        if numeric_part == "":
            numeric_part = "1"
        freq_part = match.group(2)

        if freq_part not in cls._CODES_TO_UNITS:
            raise ValueError(f"Cannot interpret frequency str: {freqstr}")

        return cls(**{cls._CODES_TO_UNITS[freq_part]: int(numeric_part)})

    @classmethod
    def _from_pandas_ticks_or_weeks(
        cls,
        tick: Union[pd.tseries.offsets.Tick, pd.tseries.offsets.Week],
    ) -> Self:
        return cls(**{cls._TICK_OR_WEEK_TO_UNITS[type(tick)]: tick.n})

    def _maybe_as_fast_pandas_offset(self):
        if (
            len(self.kwds) == 1
            and _has_fixed_frequency(self)
            and not _has_non_fixed_frequency(self)
        ):
            # Pandas computation between `n*offsets.Minute()` is faster than
            # `n*DateOffset`. If only single offset unit is in use, we return
            # the base offset for faster binary ops.
            return pd.tseries.frequencies.to_offset(pd.Timedelta(**self.kwds))
        return pd.DateOffset(**self.kwds, n=1)


def _isin_datetimelike(
    lhs: Union[column.TimeDeltaColumn, column.DatetimeColumn], values: Sequence
) -> column.ColumnBase:
    """
    Check whether values are contained in the
    DateTimeColumn or TimeDeltaColumn.

    Parameters
    ----------
    lhs : TimeDeltaColumn or DatetimeColumn
        Column to check whether the `values` exist in.
    values : set or list-like
        The sequence of values to test. Passing in a single string will
        raise a TypeError. Instead, turn a single string into a list
        of one element.

    Returns
    -------
    result: Column
        Column of booleans indicating if each element is in values.
    """
    rhs = None
    try:
        rhs = cudf.core.column.as_column(values)
        was_string = len(rhs) and rhs.dtype.kind == "O"

        if rhs.dtype.kind in {"f", "i", "u"}:
            return column.as_column(False, length=len(lhs), dtype="bool")
        rhs = rhs.astype(lhs.dtype)
        if was_string:
            warnings.warn(
                f"The behavior of 'isin' with dtype={lhs.dtype} and "
                "castable values (e.g. strings) is deprecated. In a "
                "future version, these will not be considered matching "
                "by isin. Explicitly cast to the appropriate dtype before "
                "calling isin instead.",
                FutureWarning,
            )
        res = lhs._isin_earlystop(rhs)
        if res is not None:
            return res
    except ValueError:
        # pandas functionally returns all False when cleansing via
        # typecasting fails
        return column.as_column(False, length=len(lhs), dtype="bool")

    res = lhs._obtain_isin_result(rhs)
    return res


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name=None,
    closed: Literal["left", "right", "both", "neither"] = "both",
    *,
    unit: Optional[str] = None,
):
    """Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference
    between any two adjacent points is specified by the given frequency)
    such that they all satisfy `start` <[=] x <[=] `end`, where the first one
    and the last one are, resp., the first and last time points in that range
    that are valid for `freq`.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.

    end : str or datetime-like, optional
        Right bound for generating dates.

    periods : int, optional
        Number of periods to generate.

    freq : str or DateOffset
        Frequencies to generate the datetime series. Mixed fixed-frequency and
        non-fixed frequency offset is unsupported. See notes for detail.
        Supported offset alias: ``D``, ``h``, ``H``, ``T``, ``min``, ``S``,
        ``U``, ``us``, ``N``, ``ns``.

    tz : str or tzinfo, optional
        Not Supported

    normalize : bool, default False
        Not Supported

    name : str, default None
        Name of the resulting DatetimeIndex

    closed : {"left", "right", "both", "neither"}, default "both"
        Whether to set each bound as closed or open.
        Currently only "both" is supported

    unit : str, default None
        Specify the desired resolution of the result. Currently
        not supported.

    Returns
    -------
    DatetimeIndex

    Notes
    -----
    Of the four parameters `start`, `end`, `periods`, and `freq`, exactly three
    must be specified. If `freq` is omitted, the resulting DatetimeIndex will
    have periods linearly spaced elements between start and end (closed on both
    sides).

    cudf supports `freq` specified with either fixed-frequency offset
    (such as weeks, days, hours, minutes...) or non-fixed frequency offset
    (such as years and months). Specifying `freq` with a mixed fixed and
    non-fixed frequency is currently unsupported. For example:

    >>> cudf.date_range(
    ...     start='2021-08-23 08:00:00',
    ...     freq=cudf.DateOffset(months=2, days=5),
    ...     periods=5)
    ...
    NotImplementedError: Mixing fixed and non-fixed frequency offset is
    unsupported.

    Examples
    --------
    >>> cudf.date_range(
    ...     start='2021-08-23 08:00:00',
    ...     freq=cudf.DateOffset(years=1, months=2),
    ...     periods=5)
    DatetimeIndex(['2021-08-23 08:00:00', '2022-10-23 08:00:00',
                '2023-12-23 08:00:00', '2025-02-23 08:00:00',
                '2026-04-23 08:00:00'],
                dtype='datetime64[ns]')
    """
    if closed != "both":
        raise NotImplementedError(f"{closed=} is currently unsupported.")
    if unit is not None:
        raise NotImplementedError(f"{unit=} is currently unsupported.")
    if normalize is not False:
        raise NotImplementedError(f"{normalize=} is currently unsupported.")

    if freq is None and any(arg is None for arg in (start, end, periods)):
        freq = "D"

    if (start, end, periods, freq).count(None) > 1:
        raise ValueError(
            "Of the four parameters: start, end, periods, and freq, exactly "
            "three must be specified"
        )

    if periods is not None and not cudf.api.types.is_integer(periods):
        warnings.warn(
            "Non-integer 'periods' in cudf.date_range, and cudf.interval_range"
            " are deprecated and will raise in a future version.",
            FutureWarning,
        )

    dtype = np.dtype("datetime64[ns]")

    if freq is None:
        # `start`, `end`, `periods` is specified, we treat the timestamps as
        # integers and divide the number range evenly with `periods` elements.
        start = cudf.Scalar(start, dtype=dtype).value.astype("int64")
        end = cudf.Scalar(end, dtype=dtype).value.astype("int64")
        arr = cp.linspace(start=start, stop=end, num=periods)
        result = cudf.core.column.as_column(arr).astype("datetime64[ns]")
        return cudf.DatetimeIndex._from_data({name: result}).tz_localize(tz)

    # The code logic below assumes `freq` is defined. It is first normalized
    # into `DateOffset` for further computation with timestamps.

    if isinstance(freq, DateOffset):
        offset = freq
    elif isinstance(freq, str):
        offset = pd.tseries.frequencies.to_offset(freq)
        if not isinstance(
            offset, (pd.tseries.offsets.Tick, pd.tseries.offsets.Week)
        ):
            raise ValueError(
                f"Unrecognized frequency string {freq}. cuDF does "
                "not yet support month, quarter, year-anchored frequency."
            )
        offset = DateOffset._from_pandas_ticks_or_weeks(offset)
    else:
        raise TypeError("`freq` must be a `str` or cudf.DateOffset object.")

    if _has_fixed_frequency(offset) and _has_non_fixed_frequency(offset):
        raise NotImplementedError(
            "Mixing fixed and non-fixed frequency offset is unsupported."
        )

    # Depending on different combinations of `start`, `end`, `offset`,
    # `periods`, the following logic makes sure before computing the sequence,
    # `start`, `periods`, `offset` is defined

    _periods_not_specified = False

    if start is None:
        end = cudf.Scalar(end, dtype=dtype)
        start = cudf.Scalar(
            pd.Timestamp(end.value)
            - (periods - 1) * offset._maybe_as_fast_pandas_offset(),
            dtype=dtype,
        )
    elif end is None:
        start = cudf.Scalar(start, dtype=dtype)
    elif periods is None:
        # When `periods` is unspecified, its upper bound estimated by
        # dividing the number of nanoseconds between two timestamps with
        # the lower bound of `freq` in nanoseconds. While the final result
        # may contain extra elements that exceeds `end`, they are trimmed
        # as a post processing step. [1]
        _periods_not_specified = True
        start = cudf.Scalar(start, dtype=dtype)
        end = cudf.Scalar(end, dtype=dtype)
        _is_increment_sequence = end >= start

        periods = math.ceil(
            int(end - start) / _offset_to_nanoseconds_lower_bound(offset)
        )

        if periods < 0:
            # Mismatched sign between (end-start) and offset, return empty
            # column
            periods = 0
        elif periods == 0:
            # end == start, return exactly 1 timestamp (start)
            periods = 1

    # We compute `end_estim` (the estimated upper bound of the date
    # range) below, but don't always use it.  We do this to ensure
    # that the appropriate OverflowError is raised by Pandas in case
    # of overflow.
    # FIXME: when `end_estim` is out of bound, but the actual `end` is not,
    # we shouldn't raise but compute the sequence as is. The trailing overflow
    # part should get trimmed at the end.
    with warnings.catch_warnings():
        # Need to ignore userwarnings where nonzero nanoseconds
        # are dropped in conversion during the binops
        warnings.simplefilter("ignore", UserWarning)
        end_estim = (
            pd.Timestamp(start.value)
            + periods * offset._maybe_as_fast_pandas_offset()
        ).to_datetime64()

    if "months" in offset.kwds or "years" in offset.kwds:
        # If `offset` is non-fixed frequency, resort to libcudf.
        res = libcudf.datetime.date_range(start.device_value, periods, offset)
        if _periods_not_specified:
            # As mentioned in [1], this is a post processing step to trim extra
            # elements when `periods` is an estimated value. Only offset
            # specified with non fixed frequencies requires trimming.
            res = res.apply_boolean_mask(
                (res <= end) if _is_increment_sequence else (res <= start)
            )
    else:
        # If `offset` is fixed frequency, we generate a range of
        # treating `start`, `stop` and `step` as ints:
        stop = end_estim.astype("int64")
        start = start.value.astype("int64")
        step = _offset_to_nanoseconds_lower_bound(offset)
        arr = cp.arange(start=start, stop=stop, step=step, dtype="int64")
        res = cudf.core.column.as_column(arr).astype("datetime64[ns]")

    return cudf.DatetimeIndex._from_data({name: res}, freq=freq).tz_localize(
        tz
    )


def _has_fixed_frequency(freq: DateOffset) -> bool:
    """Utility to determine if `freq` contains fixed frequency offset"""
    fixed_frequencies = {
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    }

    return len(freq.kwds.keys() & fixed_frequencies) > 0


def _has_non_fixed_frequency(freq: DateOffset) -> bool:
    """Utility to determine if `freq` contains non-fixed frequency offset"""
    non_fixed_frequencies = {"years", "months"}
    return len(freq.kwds.keys() & non_fixed_frequencies) > 0


def _offset_to_nanoseconds_lower_bound(offset: DateOffset) -> int:
    """Given a DateOffset, which can consist of either fixed frequency or
    non-fixed frequency offset, convert to the smallest possible fixed
    frequency offset based in nanoseconds.

    Specifically, the smallest fixed frequency conversion for {months=1}
    is 28 * nano_seconds_per_day, because 1 month contains at least 28 days.
    Similarly, the smallest fixed frequency conversion for {year=1} is
    365 * nano_seconds_per_day.

    This utility is used to compute the upper bound of the count of timestamps
    given a range of datetime and an offset.
    """
    nanoseconds_per_day = 24 * 60 * 60 * 10**9
    kwds = offset.kwds
    return (
        kwds.get("years", 0) * (365 * nanoseconds_per_day)
        + kwds.get("months", 0) * (28 * nanoseconds_per_day)
        + kwds.get("weeks", 0) * (7 * nanoseconds_per_day)
        + kwds.get("days", 0) * nanoseconds_per_day
        + kwds.get("hours", 0) * 3600 * 10**9
        + kwds.get("minutes", 0) * 60 * 10**9
        + kwds.get("seconds", 0) * 10**9
        + kwds.get("milliseconds", 0) * 10**6
        + kwds.get("microseconds", 0) * 10**3
        + kwds.get("nanoseconds", 0)
    )


def _to_iso_calendar(arg):
    formats = ["%G", "%V", "%u"]
    if not isinstance(arg, (cudf.Index, cudf.core.series.DatetimeProperties)):
        raise AttributeError(
            "Can only use .isocalendar accessor with series or index"
        )
    if isinstance(arg, cudf.Index):
        iso_params = [
            arg._column.as_string_column(arg.dtype, fmt) for fmt in formats
        ]
        index = arg._column
    elif isinstance(arg.series, cudf.Series):
        iso_params = [arg.strftime(fmt) for fmt in formats]
        index = arg.series.index

    data = dict(zip(["year", "week", "day"], iso_params))
    return cudf.DataFrame(data, index=index, dtype=np.int32)
