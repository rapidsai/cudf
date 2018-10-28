
import numpy as np
from pygdf.dummyarray import Array

test_slices = [
    slice(None, None, None),
    slice(None, 3, None),
    slice(3, None, None),
    slice(None, None, 3),
    slice(1, 5, None),
    slice(1, None, 2),
    slice(None, 5, 2),
    slice(1, 5, 2),
]

test_indices = [
    1, 4
]


def test_1d():
    a = np.arange(10)
    d = Array.fromarrayinterface(a)
    assert str(d) == ('Array(shape=(10,), strides=None,'
                      ' dtype="int64", offset=0)')
    assert str(d[1::3]) == ('Array(shape=(3,), strides=(24,),'
                            ' dtype="int64", offset=8)')

    for s in test_slices:
        assert Array.fromarrayinterface(a[s]) == d[s]


def test_2d():
    a = np.array([[1, 2, 3]*2, [4, 5, 6]*2]*3)

    d = Array.fromarrayinterface(a)
    assert str(d) == ('Array(shape=(6, 6), strides=None,'
                      ' dtype="int64", offset=0)')
    assert str(d[1::2, 1::2]) == ('Array(shape=(3, 3), strides=(96, 16),'
                                  ' dtype="int64", offset=56)')

    for s in test_slices:
        assert (Array.fromarrayinterface(a[s])) == (d[s])
    for s in test_indices:
        assert (Array.fromarrayinterface(a[s])) == (d[s])

    for s1 in test_slices:
        for s2 in test_slices:
            assert (Array.fromarrayinterface(a[s1, s2])) == (d[s1, s2])

    for s1 in test_indices:
        for s2 in test_slices:
            assert (Array.fromarrayinterface(a[s1, s2])) == (d[s1, s2])
            assert (Array.fromarrayinterface(a[s2, s1])) == (d[s2, s1])


def test_3d():
    a = np.arange(210).reshape((5, 6, 7))

    d = Array.fromarrayinterface(a)
    assert str(d) == ('Array(shape=(5, 6, 7), strides=None,'
                      ' dtype="int64", offset=0)')
    assert str(d[1::2, 1::2, 1::2])\
        == ('Array(shape=(2, 3, 3), strides=(672, 112, 16),'
            ' dtype="int64", offset=400)')

    for s1 in test_slices:
        assert (Array.fromarrayinterface(a[s1])) == (d[s1])
        for s2 in test_slices:
            assert (Array.fromarrayinterface(a[s1, s2])) == (d[s1, s2])
            for s3 in test_slices:
                assert (Array.fromarrayinterface(a[s1, s2, s3])) \
                    == (d[s1, s2, s3])
        for i1 in test_indices:
            assert (Array.fromarrayinterface(a[i1, s1])) == (d[i1, s1])
            assert (Array.fromarrayinterface(a[s1, i1])) == (d[s1, i1])
            for s2 in test_slices:
                assert (Array.fromarrayinterface(a[i1, s1, s2])) \
                    == (d[i1, s1, s2])
                assert (Array.fromarrayinterface(a[s1, i1, s2])) \
                    == (d[s1, i1, s2])
                assert (Array.fromarrayinterface(a[s1, s2, i1])) \
                    == (d[s1, s2, i1])
            for i2 in test_indices:
                assert (Array.fromarrayinterface(a[i1, i2])) \
                    == (d[i1, i2])
                assert (Array.fromarrayinterface(a[i1, i2, s1])) \
                    == (d[i1, i2, s1])
                assert (Array.fromarrayinterface(a[i1, s1, i2])) \
                    == (d[i1, s1, i2])
                assert (Array.fromarrayinterface(a[s1, i1, i2])) \
                    == (d[s1, i1, i2])
