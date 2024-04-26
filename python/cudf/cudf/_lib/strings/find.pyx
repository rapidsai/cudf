# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import cudf._lib.pylibcudf as plc
from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column
from cudf._lib.cpp.types cimport size_type


@acquire_spill_lock()
def contains(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain the pattern given in `py_target`.
    """
    return Column.from_pylibcudf(
        plc.strings.find.contains(
            source_strings.to_pylibcudf(mode="read"),
            py_target.device_value.c_value
        )
    )


@acquire_spill_lock()
def contains_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain the corresponding string in `target_strings`.
    """
    return Column.from_pylibcudf(
        plc.strings.find.contains(
            source_strings.to_pylibcudf(mode="read"),
            target_strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def endswith(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that end with the pattern given in `py_target`.
    """

    return Column.from_pylibcudf(
        plc.strings.find.ends_with(
            source_strings.to_pylibcudf(mode="read"),
            py_target.device_value.c_value
        )
    )


@acquire_spill_lock()
def endswith_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that end with corresponding location
    in `target_strings`.
    """
    return Column.from_pylibcudf(
        plc.strings.find.ends_with(
            source_strings.to_pylibcudf(mode="read"),
            target_strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def startswith(Column source_strings, object py_target):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that start with the pattern given in `py_target`.
    """
    return Column.from_pylibcudf(
        plc.strings.find.starts_with(
            source_strings.to_pylibcudf(mode="read"),
            py_target.device_value.c_value
        )
    )


@acquire_spill_lock()
def startswith_multiple(Column source_strings, Column target_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain strings that begin with corresponding location
    in `target_strings`.
    """
    return Column.from_pylibcudf(
        plc.strings.find.starts_with(
            source_strings.to_pylibcudf(mode="read"),
            target_strings.to_pylibcudf(mode="read")
        )
    )


@acquire_spill_lock()
def find(Column source_strings,
         object py_target,
         size_type start,
         size_type end):
    """
    Returns a Column containing lowest indexes in each string of
    `source_strings` that fully contain `py_target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
    """
    return Column.from_pylibcudf(
        plc.strings.find.find(
            source_strings.to_pylibcudf(mode="read"),
            py_target.device_value.c_value,
            start,
            end
        )
    )


@acquire_spill_lock()
def rfind(Column source_strings,
          object py_target,
          size_type start,
          size_type end):
    """
    Returns a Column containing highest indexes in each string of
    `source_strings` that fully contain `py_target` string.
    Scan portion of strings in `source_strings` can be
    controlled by setting `start` and `end` values.
    """

    return Column.from_pylibcudf(
        plc.strings.find.rfind(
            source_strings.to_pylibcudf(mode="read"),
            py_target.device_value.c_value,
            start,
            end
        )
    )
