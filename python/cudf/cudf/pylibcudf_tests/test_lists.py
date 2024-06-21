# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pytest
from utils import assert_column_eq

from cudf._lib import pylibcudf as plc


def test_concatenate_rows():
    test_data = [[[0, 1], [2], [5], [6, 7]], [[8], [9], [], [13, 14, 15]]]

    arrow_tbl = pa.Table.from_arrays(test_data, names=["a", "b"])
    plc_tbl = plc.interop.from_arrow(arrow_tbl)

    res = plc.lists.concatenate_rows(plc_tbl)

    expect = pa.array([pair[0] + pair[1] for pair in zip(*test_data)])

    assert_column_eq(expect, res)


@pytest.mark.parametrize(
    "test_data, dropna, expected",
    [
        (
            [[[1, 2], [3, 4], [5]], [[6], None, [7, 8, 9]]],
            False,
            [[1, 2, 3, 4, 5], None],
        ),
        (
            [[[1, 2], [3, 4], [5, None]], [[6], [None], [7, 8, 9]]],
            True,
            [[1, 2, 3, 4, 5, None], [6, None, 7, 8, 9]],
        ),
    ],
)
def test_concatenate_list_elements(test_data, dropna, expected):
    arr = pa.array(test_data)
    plc_column = plc.interop.from_arrow(arr)

    res = plc.lists.concatenate_list_elements(plc_column, dropna)

    expect = pa.array(expected)

    assert_column_eq(expect, res)


# @pytest.mark.parametrize(
#     "test_data, dropna, expected",
#     [
#         (
#             [[[1, 2], [3, 4], [5]], [[6], None, [7, 8, 9]]],
#             False,
#             [[1, 2, 3, 4, 5], None],
#         ),
#         (
#             [[[1, 2], [3, 4], [5, None]], [[6], [None], [7, 8, 9]]],
#             True,
#             [[1, 2, 3, 4, 5, None], [6, None, 7, 8, 9]],
#         ),
#     ],
# )
# def test_extract_list_element(test_data, dropna, expected):
#     arr = pa.array(test_data)
#     plc_column = plc.interop.from_arrow(arr)

#     res = plc.lists.concatenate_list_elements(plc_column, dropna)

#     expect = pa.array(expected)

#     assert_column_eq(expect, res)
