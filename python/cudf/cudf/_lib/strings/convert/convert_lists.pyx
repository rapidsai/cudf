# Copyright (c) 2021-2024, NVIDIA CORPORATION.

import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar


@acquire_spill_lock()
def format_list_column(Column source_list, Column separators):
    """
    Format a list column of strings into a strings column.

    Parameters
    ----------
    input_col : input column of type list with strings child.

    separators: strings used for formatting (', ', '[', ']')

    Returns
    -------
    Formatted strings column
    """
    plc_column = plc.strings.convert.convert_lists.format_list_column(
        source_list.to_pylibcudf(mode="read"),
        as_device_scalar("None").c_value,
        separators.to_pylibcudf(mode="read"),
    )
    return Column.from_pylibcudf(plc_column)
