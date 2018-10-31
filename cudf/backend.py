"""Support different CUDA backends: numba.cuda, pyarrow.cuda, librmm.

Requirements:

  arrow version 0.12+ and pyarrow.cuda availability

Usage:

  from cudf.backend import cuda
  cuda.use_numba()    # to enable numba.cuda
  cuda.use_arrow()    # to enable pyarrow.cuda, with fallback to numba.cuda
  cuda.use_rmm()      # to enable librmm

or define environment variable CUDF_BACKEND:

  CUDF_BACKEND=numba  # to enable numba.cuda
  CUDF_BACKEND=arrow  # to enable pyarrow.cuda, with fallback to numba.cuda
  CUDF_BACKEND=rmm    # to enable librmm

"""
# Author: Pearu Peterson
# Created: October 2018

import os
import numpy as np
from .cudaarray import CudaNDArray


class PyArrowCuda:
    """Provides numba.cuda and librmm api using pyarrow.cuda.
    """

    class NumbaCudaDeviceArray:

        def __init__(self, cuda):
            self.cuda = cuda

        def is_cuda_ndarray(self, obj):
            return hasattr(obj, '__cuda_array_interface__')

        def DeviceNDArray(self, *args, **kwargs):
            gpu_data = kwargs.pop('gpu_data', None)
            if gpu_data is not None:
                kwargs['cuda_data'] = self.cuda.CudaBuffer.from_numba(gpu_data)
            return CudaNDArray(*args, **kwargs)

    class NumbaCudaDriver:

        def __init__(self, cuda):
            self.cuda = cuda
            import numba.cuda
            self.__module = numba.cuda.driver

        def MemoryPointer(self, *args, **kwargs):
            return self.__module.MemoryPointer(*args, **kwargs)

    def __init__(self, parent):
        self.parent = parent
        import pyarrow
        import pyarrow.cuda
        if tuple(map(int, pyarrow.__version__.split('.')[:2])) < (0, 12):
            raise ImportError('Too old arrow version: %s. 0.12+ is required'
                              % (pyarrow.__version__))
        self.cuda = pyarrow.cuda
        self.devicearray = type(self).NumbaCudaDeviceArray(self.cuda)
        self.driver = type(self).NumbaCudaDriver(self.cuda)

    # numba.cuda API

    def current_context(self):  # used for MemoryPointer
        import numba.cuda
        return numba.cuda.current_context()

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

    # librmm API

    def device_array_from_ptr(self, ptr, nelem,
                              dtype=np.float, finalizer=None):
        if dtype == np.datetime64:
            dtype = np.dtype('datetime64[ms]')
        else:
            dtype = np.dtype(dtype)
        elemsize = dtype.itemsize
        datasize = elemsize * nelem
        ctx = self.cuda.Context(0)
        cuda_data = ctx.foreign_buffer(ptr, datasize)
        return CudaNDArray((nelem,), dtype=dtype, cuda_data=cuda_data)

    def _make_finalizer(self, *args):
        return (lambda: None)


class NumbaCuda:
    """Provides numba.cuda and librmm api using numba.cuda.
    """
    def __init__(self):
        import numba.cuda
        self.cuda = numba.cuda

    # librmm APi

    def device_array_from_ptr(self, ptr, nelem, dtype=np.float,
                              finalizer=None):
        import ctypes
        if dtype == np.datetime64:
            dtype = np.dtype('datetime64[ms]')
        else:
            dtype = np.dtype(dtype)
        elemsize = dtype.itemsize
        datasize = elemsize * nelem
        ctx = self.cuda.current_context()
        ptr = ctypes.c_uint64(ptr)
        mem = self.cuda.driver.MemoryPointer(ctx, ptr, datasize,
                                             finalizer=finalizer)
        return self.cuda.cudadrv.devicearray.DeviceNDArray((nelem,),
                                                           (elemsize,),
                                                           dtype,
                                                           gpu_data=mem)

    def _make_finalizer(self, *args):
        return (lambda: None)


class RMMCuda:
    """Provides librmm api using librmm.
    """

    def __init__(self):
        from librmm_cffi import librmm
        self.cuda = librmm
        import numba.cuda
        self.numba_cuda = numba.cuda


class CudaBackend:
    """Frontend of CUDA backend module.

    Allows switching between different cuda backends.
    """

    def __init__(self):
        backend = os.environ.get('CUDF_BACKEND', 'arrow')
        if backend == 'arrow':
            try:
                self.use_arrow()
            except ImportError as msg:
                print('Failed to enable pyarrow.cuda backend: %s.'
                      ' Using numba.cuda instead.' % (msg))
                self.use_numba()
        elif backend == 'numba':
            self.use_numba()
        elif backend == 'rmm':
            self.use_rmm()
        else:
            raise ValueError('unknown cuda backend: %r' % backend)

    def use_numba(self):
        """ Switch to using numba.cuda
        """
        self.__backend = NumbaCuda()

    def use_arrow(self):
        """ Switch to using pyarrow.cuda
        """
        self.__backend = PyArrowCuda(self)

    def use_rmm(self):
        """ Switch to using librmm
        """
        self.__backend = RMMCuda()

    def __getattr__(self, name):
        try:
            return getattr(self.__backend, name)
        except AttributeError:
            try:
                return getattr(self.__backend.cuda, name)
            except AttributeError:
                return getattr(self.__backend.numba_cuda, name)


# Initialize CUDA backend frontend.
cuda = CudaBackend()
