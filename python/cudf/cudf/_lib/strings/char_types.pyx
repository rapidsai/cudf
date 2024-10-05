# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from libcpp cimport bool

from cudf.core.buffer import acquire_spill_lock

from cudf._lib.column cimport Column

from pylibcudf.strings import char_types


@acquire_spill_lock()
def filter_alphanum(Column source_strings, object py_repl, bool keep=True):
    """
    Returns a Column of strings keeping only alphanumeric character types.
    """
    plc_column = char_types.filter_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.ALL_TYPES if keep
        else char_types.StringCharacterTypes.ALPHANUM,
        py_repl.device_value.c_value,
        char_types.StringCharacterTypes.ALPHANUM if keep
        else char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_decimal(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal characters -- those that can be used
    to extract base10 numbers.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.DECIMAL,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_alnum(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only alphanumeric characters.

    Equivalent to: is_alpha() or is_digit() or is_numeric() or is_decimal()
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.ALPHANUM,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_alpha(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only alphabetic characters.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.ALPHA,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_digit(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only decimal and digit characters.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.DIGIT,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_numeric(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only numeric characters. These include digit and
    numeric characters.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.NUMERIC,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_upper(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only upper-case characters.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.UPPER,
        char_types.StringCharacterTypes.CASE_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_lower(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contain only lower-case characters.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.LOWER,
        char_types.StringCharacterTypes.CASE_TYPES
    )
    return Column.from_pylibcudf(plc_column)


@acquire_spill_lock()
def is_space(Column source_strings):
    """
    Returns a Column of boolean values with True for `source_strings`
    that contains all characters which are spaces only.
    """
    plc_column = char_types.all_characters_of_type(
        source_strings.to_pylibcudf(mode="read"),
        char_types.StringCharacterTypes.SPACE,
        char_types.StringCharacterTypes.ALL_TYPES
    )
    return Column.from_pylibcudf(plc_column)
