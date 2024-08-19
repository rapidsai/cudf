# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf

from cudf._lib.aggregation import make_aggregation


@acquire_spill_lock()
def rolling(Column source_column,
            Column pre_column_window,
            Column fwd_column_window,
            window,
            min_periods,
            center,
            op,
            agg_params):
    """
    Rolling on input executing operation within the given window for each row

    Parameters
    ----------
    source_column : input column on which rolling operation is executed
    pre_column_window : prior window for each element of source_column
    fwd_column_window : forward window for each element of source_column
    window : Size of the moving window, can be integer or None
    min_periods : Minimum number of observations in window required to have
                  a value (otherwise result is null)
    center : Set the labels at the center of the window
    op : operation to be executed
    agg_params : dict, parameter for the aggregation (e.g. ddof for VAR/STD)

    Returns
    -------
    A Column with rolling calculations
    """

    if window is None:
        if center:
            # TODO: we can support this even though Pandas currently does not
            raise NotImplementedError(
                "center is not implemented for offset-based windows"
            )
        pre = pre_column_window.to_pylibcudf(mode="read")
        fwd = fwd_column_window.to_pylibcudf(mode="read")
    else:
        if center:
            pre = (window // 2) + 1
            fwd = window - (pre)
        else:
            pre = window
            fwd = 0

    return Column.from_pylibcudf(
        pylibcudf.rolling.rolling_window(
            source_column.to_pylibcudf(mode="read"),
            pre,
            fwd,
            min_periods,
            make_aggregation(
                op, {'dtype': source_column.dtype} if callable(op) else agg_params
            ).c_obj,
        )
    )
