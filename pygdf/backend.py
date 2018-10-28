"""Support different CUDA backends: numba.cuda, pyarrow.cuda.

Usage:
  from pygdf.backend import cuda
  cuda.use_numba()    # to enable numba.cuda
  cuda.use_arrow()    # to enable pyarrow.cuda
"""
# Author: Pearu Peterson
# Created: October 2018

import numpy as np
from .cudaarray import CudaNDArray


class NumbaCudaDeviceArray:

    def __init__(self, parent):
        self.parent = parent
        self.cuda = parent.cuda

    def is_cuda_ndarray(self, obj):
        return hasattr(obj, '__cuda_array_interface__')

    def DeviceNDArray(self, *args, **kwargs):
        gpu_data = kwargs.pop('gpu_data', None)
        if gpu_data is not None:
            kwargs['cuda_data'] = self.cuda.CudaBuffer.from_numba(gpu_data)
        return CudaNDArray(*args, **kwargs)


class NumbaCudaDriver:

    def __init__(self, parent):
        self.parent = parent
        self.cuda = parent.cuda
        import numba.cuda
        self.__module = numba.cuda.driver

    def MemoryPointer(self, *args, **kwargs):
        return self.__module.MemoryPointer(*args, **kwargs)


class PyArrowCuda:
    """ Implements numba.cuda api using pyarrow.cuda.
    """
    def __init__(self, parent):
        self.parent = parent
        import pyarrow.cuda
        self.cuda = pyarrow.cuda
        self.devicearray = NumbaCudaDeviceArray(self)
        self.driver = NumbaCudaDriver(self)

    def current_context(self):  # used for MemoryPointer
        import numba.cuda
        return numba.cuda.current_context()
        # return self.cuda.Context()

    def _auto_device(self, *args, **kwargs):  # used in cudautils only
        return self.to_device(*args, **kwargs), None

    def to_device(self, *args, **kwargs):
        return CudaNDArray.fromarray(*args, **kwargs)

    def device_array(self, shape, dtype=np.float, strides=None,
                     order='C', stream=0):
        assert stream == 0
        import numba.cuda
        ctx = numba.cuda.current_context()
        return CudaNDArray(shape, dtype=dtype, strides=strides,
                           order=order, ctx=ctx)

    def device_array_like(self, ary, stream=0):
        return self.device_array(shape=ary.shape, dtype=ary.dtype,
                                 strides=ary.strides, stream=stream)


class NumbaCuda:
    """Implements pyarrow.cuda api using numba.cuda.

    Since pygdf uses numba.cuda currently everywhere, there would not
    much need to implement pyarrow.cuda methods.
    """
    def __init__(self):
        import numba.cuda
        self.cuda = numba.cuda


class CudaBackend:
    """Frontend of CUDA backend module.

    Allows switching between different cuda backends.
    """

    def __init__(self):
        try:
            self.use_arrow()
        except ImportError as msg:
            print('Failed to enable pyarrow.cuda backend: %s'
                  ' Using numba.cuda instead.' % (msg))
            self.use_numba()

    def use_numba(self):
        """ Switch to using numba.cuda
        """
        self.__backend = NumbaCuda()

    def use_arrow(self):
        """ Switch to using pyarrow.cuda
        """
        self.__backend = PyArrowCuda(self)

    def __getattr__(self, name):
        try:
            return getattr(self.__backend, name)
        except AttributeError:
            return getattr(self.__backend.cuda, name)


cuda = CudaBackend()
# cuda.use_numba()
