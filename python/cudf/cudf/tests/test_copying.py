from __future__ import division, print_function

import numpy as np

from librmm_cffi import librmm as rmm

import cudf.bindings.copying as cpp_copying
from cudf.dataframe import columnops


def test_gather_single_col():
    col = columnops.as_column(np.arange(100), dtype=np.int32)
    gather_map = np.array([0, 1, 2, 3, 5, 8, 13, 21], dtype=np.int32)

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather(col, device_gather_map)

    np.testing.assert_array_equal(out.to_array(), gather_map)


def test_gather_cols():
    cols = [
        columnops.as_column(np.arange(10), dtype=np.int32),
        columnops.as_column(np.arange(0.0, 2.0, 0.2), dtype=np.float32),
    ]
    gather_map = np.array([0, 1, 2, 3, 5, 8], dtype=np.int32)

    expected = np.array(gather_map * 0.2, dtype=np.float32)

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather(cols, device_gather_map)

    np.testing.assert_array_equal(out[0].to_array(), gather_map)
    np.testing.assert_array_almost_equal(out[1].to_array(), expected)


def test_gather_string_col():
    col = columnops.as_column(["a", "b", "c", "d"])
    gather_map = columnops.as_column([0, 2, 3], dtype="int32").data.mem
    result = cpp_copying.apply_gather(col, gather_map)
    assert result.data.to_host() == ["a", "c", "d"]

    col = columnops.as_column(["a", "b", None, "d"])
    gather_map = columnops.as_column([0, 2, 3], dtype="int32").data.mem
    result = cpp_copying.apply_gather(col, gather_map)
    assert result.data.to_host() == ["a", None, "d"]
