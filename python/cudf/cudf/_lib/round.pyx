# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.pylibcudf.round cimport rounding_method

import cudf._lib.pylibcudf as plc


@acquire_spill_lock()
def round(Column input_col, int decimal_places=0, how="half_even"):
    """
    Round column values to the given number of decimal places

    Parameters
    ----------
    input_col : Column whose values will be rounded
    decimal_places : The number or decimal places to round to

    Returns
    -------
    A Column with values rounded to the given number of decimal places
    """
    if how not in {"half_even", "half_up"}:
        raise ValueError("'how' must be either 'half_even' or 'half_up'")

    cdef rounding_method c_how = (
        rounding_method.HALF_EVEN if how == "half_even"
        else rounding_method.HALF_UP
    )

    return Column.from_pylibcudf(
        plc.round.round(
            input_col.to_pylibcudf(mode="read"),
            decimal_places,
            c_how
        )
    )
