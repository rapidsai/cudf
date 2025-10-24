# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import itertools
import warnings
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer

import pylibcudf as plc

from cudf.api.types import is_integer, is_number, is_scalar
from cudf.core._internals import aggregation
from cudf.core.buffer import acquire_spill_lock
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.copy_types import GatherMap
from cudf.core.mixins import GetAttrGetItemMixin, Reducible
from cudf.core.multiindex import MultiIndex
from cudf.options import get_option
from cudf.utils.dtypes import SIZE_TYPE_DTYPE

if TYPE_CHECKING:
    from typing_extensions import Self

    from cudf.core.dataframe import DataFrame
    from cudf.core.index import Index
    from cudf.core.series import Series

WindowType = TypeVar("WindowType", int, plc.Column)
WindowTypePair = tuple[WindowType, WindowType]


class _RollingBase:
    """
    Contains routines to apply a window aggregation to a column.
    """

    obj: DataFrame | Series

    def _apply_agg_column(
        self, source_column: ColumnBase, agg_name: str
    ) -> ColumnBase:
        raise NotImplementedError

    def _apply_agg(self, agg_name: str, **agg_kwargs) -> DataFrame | Series:
        applied = (
            self._apply_agg_column(col, agg_name, **agg_kwargs)
            for col in self.obj._columns
        )
        return self.obj._from_data_like_self(
            self.obj._data._from_columns_like_self(applied)
        )


class Rolling(GetAttrGetItemMixin, _RollingBase, Reducible):
    """
    Rolling window calculations.

    Parameters
    ----------
    window : int, offset or a BaseIndexer subclass
        Size of the window, i.e., the number of observations used
        to calculate the statistic.
        For datetime indexes, an offset can be provided instead
        of an int. The offset must be convertible to a timedelta.
        As opposed to a fixed window size, each window will be
        sized to accommodate observations within the time period
        specified by the offset.
        If a BaseIndexer subclass is passed, calculates the window
        boundaries based on the defined ``get_window_bounds`` method.
    min_periods : int, optional
        The minimum number of observations in the window that are
        required to be non-null, so that the result is non-null.
        If not provided or ``None``, ``min_periods`` is equal to
        the window size.
    center : bool, optional
        If ``True``, the result is set at the center of the window.
        If ``False`` (default), the result is set at the right edge
        of the window.

    Returns
    -------
    ``Rolling`` object.

    Examples
    --------
    >>> import cudf
    >>> a = cudf.Series([1, 2, 3, None, 4])

    Rolling sum with window size 2.

    >>> print(a.rolling(2).sum())
    0
    1    3
    2    5
    3
    4
    dtype: int64

    Rolling sum with window size 2 and min_periods 1.

    >>> print(a.rolling(2, min_periods=1).sum())
    0    1
    1    3
    2    5
    3    3
    4    4
    dtype: int64

    Rolling count with window size 3.

    >>> print(a.rolling(3).count())
    0    1
    1    2
    2    3
    3    2
    4    2
    dtype: int64

    Rolling count with window size 3, but with the result set at the
    center of the window.

    >>> print(a.rolling(3, center=True).count())
    0    2
    1    3
    2    2
    3    2
    4    1 dtype: int64

    Rolling max with variable window size specified by an offset;
    only valid for datetime index.

    >>> a = cudf.Series(
    ...     [1, 9, 5, 4, np.nan, 1],
    ...     index=[
    ...         pd.Timestamp('20190101 09:00:00'),
    ...         pd.Timestamp('20190101 09:00:01'),
    ...         pd.Timestamp('20190101 09:00:02'),
    ...         pd.Timestamp('20190101 09:00:04'),
    ...         pd.Timestamp('20190101 09:00:07'),
    ...         pd.Timestamp('20190101 09:00:08')
    ...     ]
    ... )

    >>> print(a.rolling('2s').max())
    2019-01-01T09:00:00.000    1
    2019-01-01T09:00:01.000    9
    2019-01-01T09:00:02.000    9
    2019-01-01T09:00:04.000    4
    2019-01-01T09:00:07.000
    2019-01-01T09:00:08.000    1
    dtype: int64

    Apply custom function on the window with the *apply* method

    >>> import numpy as np
    >>> import math
    >>> b = cudf.Series([16, 25, 36, 49, 64, 81], dtype=np.float64)
    >>> def some_func(A):
    ...     b = 0
    ...     for a in A:
    ...         b = b + math.sqrt(a)
    ...     return b
    ...
    >>> print(b.rolling(3, min_periods=1).apply(some_func))
    0     4.0
    1     9.0
    2    15.0
    3    18.0
    4    21.0
    5    24.0
    dtype: float64

    And this also works for window rolling set by an offset

    >>> import pandas as pd
    >>> c = cudf.Series(
    ...     [16, 25, 36, 49, 64, 81],
    ...     index=[
    ...          pd.Timestamp('20190101 09:00:00'),
    ...          pd.Timestamp('20190101 09:00:01'),
    ...          pd.Timestamp('20190101 09:00:02'),
    ...          pd.Timestamp('20190101 09:00:04'),
    ...          pd.Timestamp('20190101 09:00:07'),
    ...          pd.Timestamp('20190101 09:00:08')
    ...      ],
    ...     dtype=np.float64
    ... )
    >>> print(c.rolling('2s').apply(some_func))
    2019-01-01T09:00:00.000     4.0
    2019-01-01T09:00:01.000     9.0
    2019-01-01T09:00:02.000    11.0
    2019-01-01T09:00:04.000     7.0
    2019-01-01T09:00:07.000     8.0
    2019-01-01T09:00:08.000    17.0
    dtype: float64
    """

    _PROTECTED_KEYS = frozenset(("obj",))

    _group_keys: Index | None = None

    _VALID_REDUCTIONS = {
        "sum",
        "min",
        "max",
        "mean",
        "var",
        "std",
    }

    def __init__(
        self,
        obj: DataFrame | Series,
        window,
        min_periods=None,
        center: bool = False,
        win_type: str | None = None,
        on=None,
        axis=0,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
    ) -> None:
        if not isinstance(center, bool):
            raise ValueError("center must be a boolean")
        self.center = center
        if axis != 0:
            warnings.warn(
                "axis is deprecated with will be removed in a future version. "
                "Transpose the DataFrame first instead."
            )
            raise NotImplementedError("axis != 0 is not supported yet.")
        self.axis = axis

        if win_type is not None:
            if win_type != "boxcar":
                raise NotImplementedError(
                    "Only the default win_type 'boxcar' is currently supported"
                )
        self.win_type = win_type

        if on is not None:
            raise NotImplementedError("on is currently not supported")
        if closed not in (None, "right"):
            raise NotImplementedError("closed is currently not supported")
        if step is not None:
            raise NotImplementedError("step is currently not supported")
        if method != "single":
            raise NotImplementedError("method is currently not supported")

        if get_option("mode.pandas_compatible"):
            obj = obj.nans_to_nulls()
        self.obj = obj

        self.window, self.min_periods = self._normalize_window_and_min_periods(
            window, min_periods
        )

    def __getitem__(self, arg) -> Self:
        if isinstance(arg, tuple):
            arg = list(arg)
        return self.obj[arg].rolling(
            window=self.window,
            min_periods=self.min_periods,
            center=self.center,
        )

    @functools.cached_property
    def _plc_windows(self) -> WindowTypePair:
        """
        Return the preceding and following windows to pass into
        pylibcudf.rolling.rolling_window
        """
        if isinstance(self.window, (int, pd.Timedelta)):
            if isinstance(self.window, pd.Timedelta):
                if self.center:
                    raise NotImplementedError(
                        "center is not implemented for frequency-based windows"
                    )
                pre = self.window.value
                fwd = 0
                orderby_obj = self.obj.index._column.astype(np.dtype(np.int64))
            else:
                if self.center:
                    pre = (self.window // 2) + 1
                    fwd = self.window - pre
                else:
                    pre = self.window
                    fwd = 0
                if self._group_keys is None:
                    # If we're doing ungrouped rolling window with
                    # integer offsets, no need to create
                    # preceding/following columns.
                    return pre, fwd
                # TODO: Expose cudf::grouped_rolling_window and use
                # that instead (perhaps), or implement an equivalent
                # to make_range_windows that takes integer window
                # bounds and group keys.
                orderby_obj = as_column(range(len(self.obj)))
            if self._group_keys is not None:
                group_cols: list[plc.Column] = [
                    col.to_pylibcudf(mode="read")
                    for col in self._group_keys._columns
                ]
            else:
                group_cols = []
            group_keys = plc.Table(group_cols)
            return plc.rolling.make_range_windows(
                group_keys,
                orderby_obj.to_pylibcudf(mode="read"),
                plc.types.Order.ASCENDING,
                plc.types.NullOrder.BEFORE,
                plc.rolling.BoundedOpen(plc.Scalar.from_py(pre)),
                plc.rolling.BoundedClosed(plc.Scalar.from_py(fwd)),
            )
        elif isinstance(self.window, BaseIndexer):
            start, end = self.window.get_window_bounds(
                num_values=len(self.obj),
                min_periods=self.min_periods,
                center=self.center,
                closed=None,
                step=None,
            )
            start = as_column(start, dtype=SIZE_TYPE_DTYPE)
            end = as_column(end, dtype=SIZE_TYPE_DTYPE)

            idx = as_column(range(len(start)))
            preceding_window = (idx - start + np.int32(1)).astype(
                SIZE_TYPE_DTYPE
            )
            following_window = (end - idx - np.int32(1)).astype(
                SIZE_TYPE_DTYPE
            )
            return (
                preceding_window.to_pylibcudf(mode="read"),
                following_window.to_pylibcudf(mode="read"),
            )
        else:
            raise ValueError(
                "self.window should have been an int, BaseIndexer, or a pandas.Timedelta "
                f"not {type(self.window).__name__}"
            )

    def _apply_agg_column(
        self, source_column: ColumnBase, agg_name: str, **agg_kwargs
    ) -> ColumnBase:
        pre, fwd = self._plc_windows
        rolling_agg = aggregation.make_aggregation(
            agg_name,
            {"dtype": source_column.dtype}
            if callable(agg_name)
            else agg_kwargs,
        ).plc_obj
        with acquire_spill_lock():
            return ColumnBase.from_pylibcudf(
                plc.rolling.rolling_window(
                    source_column.to_pylibcudf(mode="read"),
                    pre,
                    fwd,
                    self.min_periods or 1,
                    rolling_agg,
                )
            )

    def _reduce(
        self,
        op: str,
        *args,
        **kwargs,
    ) -> DataFrame | Series:
        """Calculate the rolling {op}.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object.
        """
        return self._apply_agg(op)

    def var(self, ddof: int = 1) -> DataFrame | Series:
        """Calculate the rolling variance.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of
            elements.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object.
        """
        return self._apply_agg("var", ddof=ddof)

    def std(self, ddof: int = 1) -> DataFrame | Series:
        """Calculate the rolling standard deviation.

        Parameters
        ----------
        ddof : int, default 1
            Delta Degrees of Freedom.  The divisor used in calculations
            is ``N - ddof``, where ``N`` represents the number of
            elements.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object.
        """
        return self._apply_agg("std", ddof=ddof)

    def count(self) -> DataFrame | Series:
        """Calculate the rolling count of non NaN observations.

        Returns
        -------
        Series or DataFrame
            Return type is the same as the original object.
        """
        return self._apply_agg("count")

    def apply(self, func, *args, **kwargs) -> DataFrame | Series:
        """
        Calculate the rolling custom aggregation function.

        Parameters
        ----------
        func : function
            A user defined function that takes an 1D array as input
        args : tuple
            unsupported.
        kwargs
            unsupported

        See Also
        --------
        cudf.Series.apply: Apply an elementwise function to
            transform the values in the Column.

        Notes
        -----
        The supported Python features are listed in

        https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html

        with these exceptions:

        * Math functions in `cmath` are not supported since `libcudf` does not
          have complex number support and output of `cmath` functions are most
          likely complex numbers.

        * These five functions in `math` are not supported since numba
          generates multiple PTX functions from them:

          * math.sin()
          * math.cos()
          * math.tan()
          * math.gamma()
          * math.lgamma()

        * Series with string dtypes are not supported.

        * Global variables need to be re-defined explicitly inside
          the udf, as numba considers them to be compile-time constants
          and there is no known way to obtain value of the global variable.

        Examples
        --------
        >>> import cudf
        >>> def count_if_gt_3(window):
        ...     count = 0
        ...     for i in window:
        ...             if i > 3:
        ...                     count += 1
        ...     return count
        ...
        >>> s = cudf.Series([0, 1.1, 5.8, 3.1, 6.2, 2.0, 1.5])
        >>> s.rolling(3, min_periods=1).apply(count_if_gt_3)
        0    0
        1    0
        2    1
        3    2
        4    3
        5    2
        6    1
        dtype: int64
        """
        if any(col.has_nulls() for col in self.obj._columns):
            raise NotImplementedError(
                "Handling UDF with null values is not yet supported"
            )
        return self._apply_agg(func)

    def _normalize_window_and_min_periods(
        self, window, min_periods
    ) -> tuple[int | pd.Timedelta | BaseIndexer, int | None]:
        """
        Normalize the *window* and *min_periods* args

        *window* can be:

        * An integer, in which case it is the window size.
          If *min_periods* is unspecified, it is set to be equal to
          the window size.

        * A timedelta offset, in which case it is used to generate
          a column of window sizes to use for each element.
          If *min_periods* is unspecified, it is set to 1.
          Only valid for datetime index.
        """
        if is_number(window):
            # only allow integers
            if not is_integer(window):
                raise ValueError("window must be an integer")
            if window <= 0:
                raise ValueError("window cannot be zero or negative")
            if min_periods is None:
                return window, window
            elif not is_integer(min_periods):
                raise ValueError("min_periods must be an integer")
            else:
                return window, min_periods
        elif isinstance(window, BaseIndexer):
            return window, min_periods
        elif is_scalar(window):
            try:
                window = pd.to_timedelta(window)
            except ValueError as e:
                raise ValueError(
                    "window must be integer, BaseIndexer, or convertible to a timedelta"
                ) from e
            if self.obj.index.dtype.kind != "M":
                raise ValueError(
                    "index must be a DatetimeIndex for a frequency-based window"
                )
            if min_periods is None:
                return window, 1
            else:
                return window, min_periods
        else:
            raise ValueError(
                "window must be integer, BaseIndexer, or convertible to a timedelta"
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__} [window={self.window},min_periods={self.min_periods},center={self.center}]"


class RollingGroupby(Rolling):
    """
    Grouped rolling window calculation.

    See Also
    --------
    cudf.core.window.Rolling
    """

    def __init__(self, groupby, window, min_periods=None, center=False):
        sort_order = GatherMap.from_column_unchecked(
            groupby.grouping.keys._get_sorted_inds(),
            len(groupby.obj),
            nullify=False,
        )

        # TODO: there may be overlap between the columns
        # of `groupby.grouping.keys` and `groupby.obj`.
        # As an optimization, avoid gathering those twice.
        # TODO: Unify Index._gather interface with that of IndexedFrame._gather
        self._group_keys = groupby.grouping.keys._gather(
            sort_order.column, nullify=False, check_bounds=False
        )
        obj = groupby.obj.drop(
            columns=groupby.grouping._named_columns
        )._gather(sort_order)

        super().__init__(obj, window, min_periods=min_periods, center=center)

    def _apply_agg(self, agg_name: str, **agg_kwargs) -> DataFrame | Series:
        index = MultiIndex._from_data(
            dict(
                enumerate(
                    itertools.chain(
                        self._group_keys._columns,  # type: ignore[union-attr]
                        self.obj.index._columns,
                    )
                )
            )
        )
        index.names = list(
            itertools.chain(
                self._group_keys._column_names,  # type: ignore[union-attr]
                self.obj.index._column_names,
            )
        )
        result = super()._apply_agg(agg_name, **agg_kwargs)
        result.index = index
        return result
