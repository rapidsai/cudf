import pytest

import numpy as np

from numba import cuda

from pycudf.dataframe import Buffer, BufferSentryError, Series, DataFrame


def test_buffer_basic():
    n = 10
    buf = Buffer(np.arange(n, dtype=np.float64))
    assert buf.size == n
    assert buf.capacity == n
    np.testing.assert_equal(buf.mem.copy_to_host(),
                            np.arange(n, dtype=np.float64))


def test_buffer_append():
    n = 10
    expected = np.arange(n, dtype=np.float64)
    buf = Buffer(expected, size=n - 4, capacity=n)
    assert buf.size == n - 4
    assert buf.capacity == n
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    np.testing.assert_equal(buf.as_array(), np.arange(n - 4, dtype=np.float64))

    # Buffer.append
    buf.append(1.23)
    expected[n - 4] = 1.23
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 3
    assert buf.capacity == n

    # Buffer.extend
    buf.extend(np.asarray([2, 3]))
    expected[n - 3] = 2
    expected[n - 2] = 3
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 1
    assert buf.capacity == n

    # Test out-of-bound
    with pytest.raises(MemoryError):
        buf.extend(np.asarray([2, 3]))
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n - 1
    assert buf.capacity == n

    # Append to last slot
    buf.append(10.125)
    expected[n - 1] = 10.125
    np.testing.assert_equal(buf.mem.copy_to_host(), expected)
    assert buf.size == n
    assert buf.capacity == n

    with pytest.raises(MemoryError):
        buf.append(987654)

    np.testing.assert_equal(buf.as_array(), expected)
    assert buf.size == n
    assert buf.capacity == n


def test_series_basic():
    # Make series from buffer
    a1 = np.arange(10, dtype=np.float64)
    series = Series.from_any(a1)
    assert len(series) == 10
    assert series.buffer_count == 1

    # Add new buffer
    a2 = np.arange(5)
    series.append(a2)
    assert len(series) == 15
    assert series.buffer_count == 2

    # Reserve space
    series.reserve(10)
    assert len(series) == 15
    assert series.buffer_count == 3

    # Ensure appending to previous buffer
    a3 = np.arange(3)
    series.append(a3)
    assert len(series) == 18
    assert series.buffer_count == 3

    # Flatten
    flatseries = series.flatten()
    assert flatseries.buffer_count == 1

    got = flatseries.as_array()
    expect = np.hstack([a1, a2, a3])
    np.testing.assert_equal(got, expect)


def test_dataframe_basic():
    np.random.seed(0)
    df = DataFrame()

    # Populate with cuda memory
    df['keys'] = cuda.to_device(np.arange(10, dtype=np.float64))
    np.testing.assert_equal(df['keys'].as_array(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = np.random.random(10)
    df['vals'] = rnd_vals
    np.testing.assert_equal(df['vals'].as_array(), rnd_vals)
    assert len(df) == 10
    assert df.columns == ('keys', 'vals')

    # Add rows
    df.add_row(keys=123, vals=321)
    assert len(df) == 11

    hkeys = np.asarray(np.arange(10, dtype=np.float64).tolist() + [123])
    hvals = np.asarray(rnd_vals.tolist() + [321])

    with pytest.raises(ValueError):
        df.as_matrix(['keys', 'vals'])

    # Flatten columns
    flatdf = df.flatten_columns()
    np.testing.assert_equal(flatdf['keys'].as_array(), hkeys)
    np.testing.assert_equal(flatdf['vals'].as_array(), hvals)

    mat = flatdf.as_matrix(['keys', 'vals'])
    print(mat)
    np.testing.assert_equal(mat, np.vstack([hkeys, hvals]))


if __name__ == '__main__':
    test_dataframe_basic()
