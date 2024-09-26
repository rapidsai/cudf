# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf._lib.column cimport Column


def make_timezone_transition_table(tzdir, tzname):
    plc_table = plc.io.timezone.make_timezone_transition_table(tzdir, tzname)
    return [Column.from_pylibcudf(col) for col in plc_table.columns()]
