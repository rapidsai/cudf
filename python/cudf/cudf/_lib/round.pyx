# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc
from pylibcudf.round import RoundingMethod


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

    how = (
        RoundingMethod.HALF_EVEN if how == "half_even"
        else RoundingMethod.HALF_UP
    )

    return Column.from_pylibcudf(
        plc.round.round(
            input_col.to_pylibcudf(mode="read"),
            decimal_places,
            how
        )
    )
