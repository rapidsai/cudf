# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest
import cudf
from cudf.testing._utils import assert_eq
import pylibcudf as plc
from pylibcudf.strings import split as plc_split

@pytest.mark.parametrize(
    "data, delimiter, index, expected",
    [
        # Tera original case (pylibcudf style)
        (["a_b_c", "d_e", "f"], "_", 1, ["b", "e", None]),
        # Extra: Index 0
        (["hello|world", "foo"], "|", 0, ["hello", "foo"]),
        # Out of bounds
        (["one_two", "three"], "_", 2, [None, None]),
        # Empty, None, no match
        ([None, "", "x_y_z", "no_delim"], "_", 1, [None, None, "y", None]),
    ]
)
def test_pylibcudf_split_part(data, delimiter, index, expected):
    plc_input = plc.Column.from_column(cudf.Series(data)._column.to_pylibcudf(mode="read"))
    plc_delim = plc.Scalar(delimiter)

    got = plc_split.split_part(plc_input, plc_delim, index)

    expect_col = cudf.Series(expected)._column
    assert_eq(got.to_column(), expect_col)
