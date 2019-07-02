from __future__ import division, print_function

import numpy as np

from cudf.dataframe import columnops
from librmm_cffi import librmm as rmm

import cudf.bindings.copying as cpp_copying


def test_gather_single_col():
    col = columnops.as_column(np.arange(100), dtype=np.int32)
    gather_map = np.array([0, 1, 2, 3, 5, 8, 13, 21], dtype=np.int32)

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather_column(col, device_gather_map)

    np.testing.assert_array_equal(out.to_array(), gather_map)


def test_gather_cols():
    cols = [columnops.as_column(np.arange(10), dtype=np.int32),
            columnops.as_column(np.arange(0.0, 2.0, 0.2), dtype=np.float32)]
    gather_map = np.array([0, 1, 2, 3, 5, 8], dtype=np.int32)

    expected = np.array(gather_map * 0.2, dtype=np.float32)

    device_gather_map = rmm.to_device(gather_map)

    out = cpp_copying.apply_gather(cols, device_gather_map)

    np.testing.assert_array_equal(out[0].to_array(), gather_map)
    np.testing.assert_array_almost_equal(out[1].to_array(), expected)


def test_scatter_single_col():
    col = columnops.as_column(np.arange(2.0, 0.0, -0.2), dtype=np.float32)
    col_out = columnops.as_column(np.arange(0.0, 100.0, 1.0), dtype=np.float32)
    scatter_map = np.arange(10, 0, -1).astype(np.int32)

    expected = np.arange(0.0, 100.0, 1.0)
    expected = [0.2*x if x <= 10.0 else x for x in expected]

    device_scatter_map = rmm.to_device(scatter_map)
    cpp_copying.apply_scatter_column(col, device_scatter_map, col_out)

    np.testing.assert_array_almost_equal(col_out.to_array(), expected)


def test_scatter_cols():
    cols = [columnops.as_column(np.arange(5.0, -0.5, -0.5), dtype=np.float32),
            columnops.as_column(np.arange(11), dtype=np.int32)]
    cols_out = [columnops.as_column(np.arange(0.0, 100.0, 1.0),
                dtype=np.float32),
                columnops.as_column(np.arange(100), dtype=np.int32)]
    scatter_map = np.arange(10, -1, -1).astype(np.int32)

    expected = np.arange(0.0, 100.0, 1.0)
    expected = [0.5*x if x <= 10.0 else x for x in expected]

    expected2 = np.arange(100)
    expected2 = [10 - x if x <= 10 else x for x in expected2]

    device_scatter_map = rmm.to_device(scatter_map)
    cpp_copying.apply_scatter(cols, device_scatter_map, cols_out)

    np.testing.assert_array_almost_equal(cols_out[0].to_array(), expected)
    np.testing.assert_array_equal(cols_out[1].to_array(), expected2)


def test_gather_string_col():
    col = columnops.as_column(['a', 'b', 'c', 'd'])
    gather_map = columnops.as_column([0, 2, 3], dtype='int32').data.mem
    result = cpp_copying.apply_gather([col], gather_map)
    assert(result[0].data.to_host() == ['a', 'c', 'd'])

    col = columnops.as_column(['a', 'b', None, 'd'])
    gather_map = columnops.as_column([0, 2, 3], dtype='int32').data.mem
    result = cpp_copying.apply_gather([col], gather_map)
    assert(result[0].data.to_host() == ['a', None, 'd'])
