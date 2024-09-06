# Copyright (c) 2024, NVIDIA CORPORATION.

import pyarrow as pa
import pylibcudf as plc
import pytest


@pytest.mark.parametrize("left_inclusive", [True, False])
@pytest.mark.parametrize("right_inclusive", [True, False])
def test_label_bins(left_inclusive, right_inclusive):
    in_col = plc.interop.from_arrow(pa.array([1, 2, 3]))
    left_edges = plc.interop.from_arrow(pa.array([0, 5]))
    right_edges = plc.interop.from_arrow(pa.array([4, 6]))
    result = plc.interop.to_arrow(
        plc.labeling.label_bins(
            in_col, left_edges, left_inclusive, right_edges, right_inclusive
        )
    )
    expected = pa.chunked_array([[0, 0, 0]], type=pa.int32())
    assert result.equals(expected)
