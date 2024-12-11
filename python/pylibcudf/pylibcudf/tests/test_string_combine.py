# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def test_concatenate_scalar_seperator():
    plc_table = plc.interop.from_arrow(
        pa.table({"a": ["a", None, "c"], "b": ["a", "b", None]})
    )
    sep = plc.interop.from_arrow(pa.scalar("-"))
    result = plc.strings.combine.concatenate(
        plc_table,
        sep,
    )
    expected = pa.array(["a-a", "-b", "c-"])
    assert_column_eq(result, expected)

    result = plc.strings.combine.concatenate(
        plc_table, sep, narep=plc.interop.from_arrow(pa.scalar("!"))
    )
    expected = pa.array(["a-a", "!-b", "c-!"])
    assert_column_eq(result, expected)

    with pytest.raises(ValueError):
        plc.strings.combine.concatenate(
            plc_table,
            sep,
            narep=plc.interop.from_arrow(pa.scalar("!")),
            col_narep=plc.interop.from_arrow(pa.scalar("?")),
        )


def test_concatenate_column_seperator():
    plc_table = plc.interop.from_arrow(
        pa.table({"a": ["a", None, "c"], "b": ["a", "b", None]})
    )
    sep = plc.interop.from_arrow(pa.array(["-", "?", ","]))
    result = plc.strings.combine.concatenate(
        plc_table,
        sep,
    )
    expected = pa.array(["a-a", "?b", "c,"])
    assert_column_eq(result, expected)

    result = plc.strings.combine.concatenate(
        plc_table,
        plc.interop.from_arrow(pa.array([None, "?", ","])),
        narep=plc.interop.from_arrow(pa.scalar("1")),
        col_narep=plc.interop.from_arrow(pa.scalar("*")),
    )
    expected = pa.array(["a1a", "*?b", "c,*"])
    assert_column_eq(result, expected)


def test_join_strings():
    pa_arr = pa.array(list("abc"))
    sep = pa.scalar("")
    result = plc.strings.combine.join_strings(
        plc.interop.from_arrow(pa_arr),
        plc.interop.from_arrow(sep),
        plc.interop.from_arrow(pa.scalar("")),
    )
    expected = pa.array(["abc"])
    assert_column_eq(result, expected)


def test_join_list_elements():
    pa_arr = pa.array([["a", "a"], ["b", "b"]])
    sep = pa.scalar("")
    result = plc.strings.combine.join_list_elements(
        plc.interop.from_arrow(pa_arr),
        plc.interop.from_arrow(sep),
        plc.interop.from_arrow(pa.scalar("")),
        plc.interop.from_arrow(pa.scalar("")),
        plc.strings.combine.SeparatorOnNulls.YES,
        plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
    )
    expected = pc.binary_join(pa.array([["a", "a"], ["b", "b"]]), sep)
    assert_column_eq(result, expected)
