import pandas as pd

import cudf
from cudf.bindings.rolling import apply_rolling
from cudf.dataframe.columnops import as_column
from cudf.utils import cudautils


class Rolling:
    """
    Rolling window calculations.

    Parameters
    ----------
    window : int
        Size of the window, i.e., the number of observations used
        to calculate the statistic. Currently cuDF only supports
        a fixed window size.
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
    """
    def __init__(self, obj, window, min_periods=None, center=False):
        self.obj = obj
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self._normalize()

    def _apply_agg_series(self, sr, agg_name):
        result_col = apply_rolling(
            sr._column,
            self._window,
            self.min_periods,
            self.center,
            agg_name)
        return sr._copy_construct(data=result_col)

    def _apply_agg_dataframe(self, df, agg_name):
        result_df = cudf.DataFrame({})
        for col_name in df.columns:
            result_col = self._apply_agg_series(df[col_name], agg_name)
            result_df.add_column(name=col_name, data=result_col)
        result_df.index = df.index
        return result_df

    def _apply_agg(self, agg_name):
        if isinstance(self.obj, cudf.Series):
            return self._apply_agg_series(self.obj, agg_name)
        else:
            return self._apply_agg_dataframe(self.obj, agg_name)

    def sum(self):
        return self._apply_agg("sum")

    def min(self):
        return self._apply_agg("min")

    def max(self):
        return self._apply_agg("max")

    def mean(self):
        return self._apply_agg("mean")

    def count(self):
        return self._apply_agg("count")

    def _normalize(self):
        self._window, self.min_periods = self._normalize_window_and_min_periods()

    def _normalize_window_and_min_periods(self):
        """
        *window* can be:

        * An integer, in which case it is the window size.
          If *min_periods* is unspecified, it is set to be equal to
          the window size.

        * A timedelta offset, in which case it is used to generate
          a column of window sizes to use for each element.
          If *min_periods* is unspecified, it is set to 1.
        """
        window, min_periods = self.window, self.min_periods
        if pd.api.types.is_number(window):
            # only allow integers
            if not pd.api.types.is_integer(window):
                raise ValueError("window must be an integer")
            if window <= 0:
                raise ValueError("window cannot be zero or negative")
            if self.min_periods is None:
                min_periods = window
        else:
            if not isinstance(self.obj.index, cudf.dataframe.index.DatetimeIndex):
                raise ValueError("window must be an integer")
            try:
                window = pd.to_timedelta(window)
                # to_timedelta will also convert np.arrays etc.,
                if not isinstance(window, pd.Timedelta):
                    raise ValueError
                window = window.to_timedelta64()
            except ValueError as e:
                raise ValueError("window must be integer or "
                                 "convertible to a timedelta") from e
            window = cudautils.window_sizes_from_offset(
                self.obj.index.as_column().data.mem,
                window
            )
            if self.min_periods is None:
                min_periods = 1
        return window, min_periods

    def __repr__(self):
        return "{} [window={},min_periods={},center={}]".format(
            self.__class__.__name__,
            self.window,
            self.min_periods,
            self.center
        )
