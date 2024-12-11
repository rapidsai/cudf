# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq, assert_table_eq

import pylibcudf as plc


@pytest.fixture
def data_col():
    pa_array = pa.array(["a_b_c", "d-e-f", None])
    plc_column = plc.interop.from_arrow(pa_array)
    return pa_array, plc_column


@pytest.fixture
def delimiter():
    delimiter = "_"
    plc_delimiter = plc.interop.from_arrow(pa.scalar(delimiter))
    return delimiter, plc_delimiter


@pytest.fixture
def re_delimiter():
    return "[_-]"


def test_split(data_col, delimiter):
    _, plc_column = data_col
    _, plc_delimiter = delimiter
    result = plc.strings.split.split.split(plc_column, plc_delimiter, 1)
    expected = pa.table(
        {
            "a": ["a", "d-e-f", None],
            "b": ["b_c", None, None],
        }
    )
    assert_table_eq(expected, result)


def test_rsplit(data_col, delimiter):
    _, plc_column = data_col
    _, plc_delimiter = delimiter
    result = plc.strings.split.split.rsplit(plc_column, plc_delimiter, 1)
    expected = pa.table(
        {
            "a": ["a_b", "d-e-f", None],
            "b": ["c", None, None],
        }
    )
    assert_table_eq(expected, result)


def test_split_record(data_col, delimiter):
    pa_array, plc_column = data_col
    delim, plc_delim = delimiter
    result = plc.strings.split.split.split_record(plc_column, plc_delim, 1)
    expected = pc.split_pattern(pa_array, delim, max_splits=1)
    assert_column_eq(expected, result)


def test_rsplit_record(data_col, delimiter):
    pa_array, plc_column = data_col
    delim, plc_delim = delimiter
    result = plc.strings.split.split.split_record(plc_column, plc_delim, 1)
    expected = pc.split_pattern(pa_array, delim, max_splits=1)
    assert_column_eq(expected, result)


def test_split_re(data_col, re_delimiter):
    _, plc_column = data_col
    result = plc.strings.split.split.split_re(
        plc_column,
        plc.strings.regex_program.RegexProgram.create(
            re_delimiter, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        1,
    )
    expected = pa.table(
        {
            "a": ["a", "d", None],
            "b": ["b_c", "e-f", None],
        }
    )
    assert_table_eq(expected, result)


def test_rsplit_re(data_col, re_delimiter):
    _, plc_column = data_col
    result = plc.strings.split.split.rsplit_re(
        plc_column,
        plc.strings.regex_program.RegexProgram.create(
            re_delimiter, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        1,
    )
    expected = pa.table(
        {
            "a": ["a_b", "d-e", None],
            "b": ["c", "f", None],
        }
    )
    assert_table_eq(expected, result)


def test_split_record_re(data_col, re_delimiter):
    pa_array, plc_column = data_col
    result = plc.strings.split.split.split_record_re(
        plc_column,
        plc.strings.regex_program.RegexProgram.create(
            re_delimiter, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        1,
    )
    expected = pc.split_pattern_regex(pa_array, re_delimiter, max_splits=1)
    assert_column_eq(expected, result)


def test_rsplit_record_re(data_col, re_delimiter):
    pa_array, plc_column = data_col
    result = plc.strings.split.split.rsplit_record_re(
        plc_column,
        plc.strings.regex_program.RegexProgram.create(
            re_delimiter, plc.strings.regex_flags.RegexFlags.DEFAULT
        ),
        -1,
    )
    expected = pc.split_pattern_regex(pa_array, re_delimiter)
    assert_column_eq(expected, result)
