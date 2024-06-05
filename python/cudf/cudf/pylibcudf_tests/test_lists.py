# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def test_concatenate_rows():
    test_data = [[[0, 1], [2], [5], [6, 7]], [[8], [9], [], [13, 14, 15]]]

    arrow_tbl = pa.Table.from_arrays(test_data, names=["a", "b"])
    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    res = plc.lists.concatenate_rows(plc_tbl)

    expect = pa.array([pair[0] + pair[1] for pair in zip(*test_data)])

    assert_column_eq(res, expect)


def test_concatenate_list_elements():
    test_data = [[[1, 2], [3, 4], [5]], [[6], [], [7, 8, 9]]]

    arr = pa.array(test_data)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.concatenate_list_elements(plc_column, False)

    expect = pa.array([[1, 2, 3, 4, 5], [6, 7, 8, 9]])

    assert_column_eq(res, expect)
