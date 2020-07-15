# Copyright (c) 2019-2020, NVIDIA CORPORATION.

import warnings

import numpy as np
from pandas.core.tools.datetimes import _unit_map

import cudf
from cudf._lib.scalar import as_scalar
from cudf._lib.strings.char_types import is_integer as cpp_is_integer
from cudf.core import column
from cudf.core.index import as_index
from cudf.utils.dtypes import is_scalar

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
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit="ns",
    infer_datetime_format=False,
    origin="unix",
    cache=True,
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
    infer_datetime_format : bool, default False
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
    if arg is None:
        return None

    if exact is False:
        raise NotImplementedError("exact support is not yet implemented")

    if origin != "unix":
        raise NotImplementedError("origin support is not yet implemented")

    if yearfirst:
        raise NotImplementedError("yearfirst support is not yet implemented")

    try:
        if isinstance(arg, cudf.DataFrame):
            # we require at least Ymd
            required = ["year", "month", "day"]
            req = list(set(required) - set(arg._data.names))
            if len(req):
                req = ",".join(req)
                raise ValueError(
                    f"to assemble mappings requires at least that "
                    f"[year, month, day] be specified: [{req}] "
                    f"is missing"
                )

            # replace passed column name with values in _unit_map
            unit = {k: get_units(k) for k in arg._data.names}
            unit_rev = {v: k for k, v in unit.items()}

            # keys we don't recognize
            excess = set(unit_rev.keys()) - set(_unit_map.values())
            if len(excess):
                excess = ",".join(excess)
                raise ValueError(
                    f"extra keys have been passed to the "
                    f"datetime assemblage: [{excess}]"
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

                    factor = as_scalar(
                        column.datetime._numpy_to_pandas_conversion[u]
                        / (
                            column.datetime._numpy_to_pandas_conversion["s"]
                            if np.datetime_data(col.dtype)[0] == "s"
                            else 1
                        )
                    )

                    if times_column is None:
                        times_column = current_col.binary_operator(
                            binop="mul", rhs=factor
                        )
                    else:
                        times_column = times_column.binary_operator(
                            binop="add",
                            rhs=current_col.binary_operator(
                                binop="mul", rhs=factor
                            ),
                        )
            if times_column is not None:
                col = (
                    col.astype(dtype="int64")
                    .binary_operator(binop="add", rhs=times_column)
                    .astype(dtype=col.dtype)
                )
            return cudf.Series(col, index=arg.index)
        elif isinstance(arg, cudf.Index):
            col = arg._values
            col = _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
            )
            return as_index(col, name=arg.name)
        elif isinstance(arg, cudf.Series):
            col = arg._column
            col = _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
            )
            return cudf.Series(col, index=arg.index, name=arg.name)
        else:
            col = column.as_column(arg)
            col = _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                infer_datetime_format=infer_datetime_format,
                format=format,
            )

            if is_scalar(arg):
                return col[0]
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


def _process_col(col, unit, dayfirst, infer_datetime_format, format):
    if col.dtype.kind == "M":
        return col

    if col.dtype.kind in ("f"):
        if unit not in (None, "ns"):
            factor = as_scalar(
                column.datetime._numpy_to_pandas_conversion[unit]
            )
            col = col.binary_operator(binop="mul", rhs=factor)
        col = col.as_datetime_column(dtype="datetime64[ns]")
    if col.dtype.kind in ("i"):
        if unit in ("D", "h", "m"):
            factor = as_scalar(
                column.datetime._numpy_to_pandas_conversion[unit]
                / column.datetime._numpy_to_pandas_conversion["s"]
            )
            col = col.binary_operator(binop="mul", rhs=factor)

        col = col.as_datetime_column(dtype=_unit_dtype_map[unit])
    elif col.dtype.kind in ("O"):
        if unit not in (None, "ns"):
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
            )
        else:
            if infer_datetime_format and format is None:
                format = column.datetime.infer_format(
                    element=col[0], dayfirst=dayfirst,
                )
            elif format is None:
                format = column.datetime.infer_format(element=col[0])
            col = col.as_datetime_column(
                dtype=_unit_dtype_map[unit], format=format,
            )
    return col


def get_units(value):
    if value in _unit_map:
        return _unit_map[value]

    # m is case significant
    if value.lower() in _unit_map:
        return _unit_map[value.lower()]

    return value
