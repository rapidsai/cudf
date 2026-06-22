# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import pytest

import cudf


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("multiindex", [True, False])
def test_searchsorted_dataframe(side, multiindex):
    values = cudf.DataFrame(
        {
            "a": [1, 0, 5, 1],
            "b": [-0.998, 0.031, -0.888, -0.998],
            "c": ["C", "A", "G", "B"],
        }
    )
    base = cudf.DataFrame(
        {
            "a": [1, 1, 1, 5],
            "b": [-0.999, -0.998, -0.997, -0.888],
            "c": ["A", "C", "E", "G"],
        }
    )

    if multiindex:
        base = base.set_index(["a", "b", "c"]).index
        values = values.set_index(["a", "b", "c"]).index

    result = base.searchsorted(values, side=side).tolist()

    if side == "left":
        assert result == [1, 0, 3, 1]
    else:
        assert result == [2, 0, 4, 1]


def test_search_sorted_dataframe_unequal_number_of_columns():
    values = cudf.DataFrame({"a": [1, 0, 5, 1]})
    base = cudf.DataFrame({"a": [1, 0, 5, 1], "b": ["x", "z", "w", "a"]})

    with pytest.raises(ValueError, match="Mismatch number of columns"):
        base.searchsorted(values)
