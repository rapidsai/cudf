# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture
def data_col():
    pa_data_col = pa.array(
        ["aa", "bbb", "cccc", "abcd", None],
        type=pa.string(),
    )
    return pa_data_col, plc.interop.from_arrow(pa_data_col)


@pytest.fixture
def trans_table():
    return str.maketrans("abd", "A Q")


def test_translate(data_col, trans_table):
    pa_array, plc_col = data_col
    result = plc.strings.translate.translate(plc_col, trans_table)
    expected = pa.array(
        [
            val.translate(trans_table) if isinstance(val, str) else None
            for val in pa_array.to_pylist()
        ]
    )
    assert_column_eq(expected, result)


@pytest.mark.parametrize(
    "keep",
    [
        plc.strings.translate.FilterType.KEEP,
        plc.strings.translate.FilterType.REMOVE,
    ],
)
def test_filter_characters(data_col, trans_table, keep):
    pa_array, plc_col = data_col
    result = plc.strings.translate.filter_characters(
        plc_col, trans_table, keep, plc.interop.from_arrow(pa.scalar("*"))
    )
    exp_data = []
    flat_trans = set(trans_table.keys()).union(trans_table.values())
    for val in pa_array.to_pylist():
        if not isinstance(val, str):
            exp_data.append(val)
        else:
            new_val = ""
            for ch in val:
                if (
                    ch in flat_trans
                    and keep == plc.strings.translate.FilterType.KEEP
                ):
                    new_val += ch
                elif (
                    ch not in flat_trans
                    and keep == plc.strings.translate.FilterType.REMOVE
                ):
                    new_val += ch
                else:
                    new_val += "*"
            exp_data.append(new_val)
    expected = pa.array(exp_data)
    assert_column_eq(expected, result)
