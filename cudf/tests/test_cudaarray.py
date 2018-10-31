import pytest
import numpy as np
from cudf.cudaarray import CudaNDArray
from .backends import cuda_backend_test

slices = [
    slice(None),
    slice(2, 5),
    slice(None, 5),
    slice(2, None),
    slice(None, None, 1),
    slice(None, None, 2),
    slice(1, None, 2),
    slice(1, 5, 2),
    slice(1, None, 3),
    slice(1, 1),
]

indices = [0, 1, 5, -1, -2]


@cuda_backend_test
def test_1d(cuda):
    arr = np.arange(10)
    carr = CudaNDArray.fromarray(arr)

    np.testing.assert_equal(carr.copy_to_host(), arr)
    for s in slices:
        np.testing.assert_equal(carr[s].copy_to_host(), arr[s])

    for i in indices:
        np.testing.assert_equal(carr[i], arr[i])


@cuda_backend_test
def test_2d(cuda):
    arr = np.arange(110).reshape((10, 11))
    carr = CudaNDArray.fromarray(arr)
    np.testing.assert_equal(carr.copy_to_host(), arr)

    for s1 in slices:
        np.testing.assert_equal(carr[s1].copy_to_host(), arr[s1])
        for i1 in indices:
            np.testing.assert_equal(carr[s1, i1].copy_to_host(), arr[s1, i1])
            np.testing.assert_equal(carr[i1, s1].copy_to_host(), arr[i1, s1])
        for s2 in slices:
            np.testing.assert_equal(carr[s1, s2].copy_to_host(), arr[s1, s2])

    for i1 in indices:
        np.testing.assert_equal(carr[i1].copy_to_host(), arr[i1])
        for i2 in indices:
            np.testing.assert_equal(carr[i1, i2], arr[i1, i2])
            np.testing.assert_equal(carr[i1][i2], arr[i1][i2])


@cuda_backend_test
def test_reshape_ravel(cuda):
    arr = np.arange(110)
    carr = CudaNDArray.fromarray(arr)

    for shapes in ((110, ), (10, 11), (5, 2, 11)):
        np.testing.assert_equal(carr.reshape(*shapes).copy_to_host(),
                                arr.reshape(*shapes))
        np.testing.assert_equal(carr.reshape(*shapes,
                                             **dict(order='F'))
                                .copy_to_host(),
                                arr.reshape(*shapes, **dict(order='F')))
        np.testing.assert_equal(carr.reshape(*shapes).ravel()
                                .copy_to_host(),
                                arr.reshape(*shapes).ravel())
        np.testing.assert_equal(carr.reshape(*shapes,
                                             **dict(order='C'))
                                .ravel(order='C')
                                .copy_to_host(),
                                arr.reshape(*shapes,
                                            **dict(order='C'))
                                .ravel(order='C'))
        np.testing.assert_equal(carr.reshape(*shapes,
                                             **dict(order='F'))
                                .ravel(order='F')
                                .copy_to_host(),
                                arr.reshape(*shapes,
                                            **dict(order='F'))
                                .ravel(order='F'))
        if len(shapes) > 1:
            with pytest.raises(NotImplementedError,
                               message="Expecting NotImplementedError"):
                carr.reshape(*shapes, **dict(order='F')).ravel(order='C')
                carr.reshape(*shapes, **dict(order='C')).ravel(order='F')


@cuda_backend_test
def test_setitem_1d(cuda):
    arr = np.arange(10)
    carr = CudaNDArray.fromarray(arr)
    rarr = arr.copy()
    carr[::2] = 13
    carr[1::2] = np.arange(5)*100
    rarr[::2] = 13
    rarr[1::2] = np.arange(5)*100
    np.testing.assert_equal(carr, rarr)


@cuda_backend_test
def test_setitem_2d(cuda):
    arr = np.arange(10).reshape(5, 2)
    carr = CudaNDArray.fromarray(arr)
    rarr = arr.copy()
    carr[2] = rarr[2] = 999
    carr[3] = rarr[3] = [1, 2]
    carr[0, 0] = rarr[0, 0] = 1000
    np.testing.assert_equal(carr, rarr)


@cuda_backend_test
def test_numba_interop(cuda):
    arr = np.arange(10)
    carr = CudaNDArray.fromarray(arr)
    narr = carr.toDeviceNDArray()
    arr2 = narr.__array__()
    np.testing.assert_equal(arr, arr2)
    carr[1] = 999
    assert narr[1] == 999
    narr[2] = 777
    assert carr[2] == 777
