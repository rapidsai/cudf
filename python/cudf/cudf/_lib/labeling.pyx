# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp cimport bool as cbool

import pylibcudf as plc

from cudf._lib.column cimport Column
from cudf.core.buffer import acquire_spill_lock


# Note that the parameter input shadows a Python built-in in the local scope,
# but I'm not too concerned about that since there's no use-case for actual
# input in this context.
@acquire_spill_lock()
def label_bins(Column input, Column left_edges, cbool left_inclusive,
               Column right_edges, cbool right_inclusive):
    plc_column = plc.labeling.label_bins(
        input.to_pylibcudf(mode="read"),
        left_edges.to_pylibcudf(mode="read"),
        left_inclusive,
        right_edges.to_pylibcudf(mode="read"),
        right_inclusive
    )
    return Column.from_pylibcudf(plc_column)
