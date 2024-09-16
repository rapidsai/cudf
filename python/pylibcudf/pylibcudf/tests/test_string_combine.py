# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pyarrow.compute as pc
import pylibcudf as plc


def test_concatenate():
    arr = pa.array(["a", "b"])
    sep = pa.scalar("")
    plc_table = plc.interop.from_arrow(pa.table({"a": arr, "b": arr}))
    plc_result = plc.strings.combine.concatenate(
        plc_table,
        plc.interop.from_arrow(sep),
        plc.interop.from_arrow(pa.scalar("")),
        plc.interop.from_arrow(pa.scalar("")),
        plc.strings.combine.SeparatorOnNulls.YES,
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(
        pc.binary_join(pa.array([["a", "a"], ["b", "b"]]), sep)
    )
    assert result.equals(expected)


def test_join_strings():
    pa_arr = pa.array(list("abc"))
    sep = pa.scalar("")
    plc_result = plc.strings.combine.join_strings(
        plc.interop.from_arrow(pa_arr),
        plc.interop.from_arrow(sep),
        plc.interop.from_arrow(pa.scalar("")),
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array([["abc"]])
    assert result.equals(expected)


def test_join_list_elements():
    pa_arr = pa.array([["a", "a"], ["b", "b"]])
    sep = pa.scalar("")
    plc_result = plc.strings.combine.join_list_elements(
        plc.interop.from_arrow(pa_arr),
        plc.interop.from_arrow(sep),
        plc.interop.from_arrow(pa.scalar("")),
        plc.interop.from_arrow(pa.scalar("")),
        plc.strings.combine.SeparatorOnNulls.YES,
        plc.strings.combine.OutputIfEmptyList.NULL_ELEMENT,
    )
    result = plc.interop.to_arrow(plc_result)
    expected = pa.chunked_array(
        pc.binary_join(pa.array([["a", "a"], ["b", "b"]]), sep)
    )
    assert result.equals(expected)
