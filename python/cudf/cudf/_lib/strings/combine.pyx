# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

import pylibcudf as plc

import cudf


@acquire_spill_lock()
def concatenate(list source_strings,
                object sep,
                object na_rep):
    """
    Returns a Column by concatenating strings column-wise in `source_strings`
    with the specified `sep` between each column and
    `na`/`None` values are replaced by `na_rep`
    """
    plc_column = plc.strings.combine.concatenate(
        plc.Table([col.to_pylibcudf(mode="read") for col in source_strings]),
        sep.device_value.c_value,
        na_rep.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def join(Column source_strings,
         object sep,
         object na_rep):
    """
    Returns a Column by concatenating strings row-wise in `source_strings`
    with the specified `sep` between each column and
    `na`/`None` values are replaced by `na_rep`
    """
    plc_column = plc.strings.combine.join_strings(
        source_strings.to_pylibcudf(mode="read"),
        sep.device_value.c_value,
        na_rep.device_value.c_value,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def join_lists_with_scalar(
        Column source_strings,
        object py_separator,
        object py_narep):
    """
    Returns a Column by concatenating Lists of strings row-wise
    in `source_strings` with the specified `py_separator`
    between each string in lists and `<NA>`/`None` values
    are replaced by `py_narep`
    """
    plc_column = plc.strings.combine.join_list_elements(
        source_strings.to_pylibcudf(mode="read"),
        py_separator.device_value.c_value,
        py_narep.device_value.c_value,
        cudf._lib.scalar.DeviceScalar("", cudf.dtype("object")).c_value,
        plc.strings.combine.SeparatorOnNulls.YES,
        plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def join_lists_with_column(
        Column source_strings,
        Column separator_strings,
        object py_source_narep,
        object py_separator_narep):
    """
    Returns a Column by concatenating Lists of strings row-wise in
    `source_strings` with a corresponding separator at the same
    position in `separator_strings` and `<NA>`/`None` values in
    `source_strings` are replaced by `py_source_narep` and
    `<NA>`/`None` values in `separator_strings` are replaced
    by `py_separator_narep`
    """
    plc_column = plc.strings.combine.join_list_elements(
        source_strings.to_pylibcudf(mode="read"),
        separator_strings.to_pylibcudf(mode="read"),
        py_separator_narep.device_value.c_value,
        py_source_narep.device_value.c_value,
        plc.strings.combine.SeparatorOnNulls.YES,
        plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
    )
    return Column.from_pylibcudf(plc_column)
