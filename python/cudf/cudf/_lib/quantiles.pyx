# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from libcpp cimport bool
from libcpp.vector cimport vector

from cudf._lib.column cimport Column

from cudf._lib.utils cimport columns_from_pylibcudf_table

import pylibcudf as plc


@acquire_spill_lock()
def quantile(
    Column input,
    vector[double] q,
    str interp,
    Column ordered_indices,
    bool exact,
):
    return Column.from_pylibcudf(
        plc.quantiles.quantile(
            input.to_pylibcudf(mode="read"),
            q,
            plc.types.Interpolation[interp.upper()],
            ordered_indices.to_pylibcudf(mode="read"),
            exact
        )
    )


def quantile_table(
    list source_columns,
    vector[double] q,
    object interp,
    object is_input_sorted,
    list column_order,
    list null_precedence,
):
    return columns_from_pylibcudf_table(
        plc.quantiles.quantiles(
            plc.Table([
                c.to_pylibcudf(mode="read") for c in source_columns
            ]),
            q,
            interp,
            is_input_sorted,
            column_order,
            null_precedence
        )
    )
