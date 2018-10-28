"""Implements CudaNDArray with the same API as
numba.cuda.DeviceNDArray but device memory is held in arrow
CudaBuffer.

"""
# Author: Pearu Peterson
# Created: September 2018

from ctypes import c_void_p
import numpy as np
import pyarrow as pa
from . import dummyarray
from numba import numpy_support, types

integer_types = (int,)


def prepare_shape_strides_dtype(shape, strides, dtype, order='C'):
    if dtype is None:
        dtype = np.uint8
    dtype = np.dtype(dtype)
    if isinstance(shape, integer_types):
        shape = shape,
    elif shape == ():
        shape = (1,)
    if isinstance(strides, integer_types):
        strides = strides,
    else:
        if strides is None:
            nd = len(shape)
            strides = [0] * nd
            if order == 'C':
                strides[-1] = dtype.itemsize
                for d in reversed(range(nd - 1)):
                    strides[d] = strides[d + 1] * shape[d + 1]
            elif order == 'F':
                strides[0] = dtype.itemsize
                for d in range(1, nd):
                    strides[d] = strides[d - 1] * shape[d - 1]
            else:
                raise ValueError('order must be "C" or "F" but got %r'
                                 % (order))
    assert len(strides) == len(shape)
    return tuple(shape), tuple(strides), dtype


class CudaNDArray(object):
    """CudaNDArray represents CUDA device memory as a multidimensional
    array that items can be accessed from host.

    The device memory is accessed via pyarrow.cuda.CudaBuffer instance.

    numba.cuda defines DeviceNDArray for the same purpose as
    CudaNDArray but DeviceNDArray accesses the device memory via
    device pointer (stored in MemoryPointer). Since numba jit
    functions use DeviceNDArray, we implement cooperativity between
    the two representations.

    """

    def __init__(self, shape, strides=None, dtype=None, cuda_data=None,
                 order='C', ctx=None):
        self.shape, self.strides, self.dtype \
            = prepare_shape_strides_dtype(shape, strides, dtype, order=order)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.alloc_size = self.size * self.dtype.itemsize
        self._dummy = dummyarray.Array(self.shape, strides=self.strides,
                                       dtype=self.dtype, order=order)
        if cuda_data is None:
            if ctx is None:
                ctx = pa.cuda.Context()
            if not isinstance(ctx, pa.cuda.Context):
                ctx = pa.cuda.Context.from_numba(ctx)
            cuda_data = ctx.new_buffer(self.alloc_size)
        elif isinstance(cuda_data, pa.cuda.CudaBuffer):
            pass
        elif isinstance(cuda_data, CudaNDArray):
            raise NotImplementedError('CUDA device array from a device array')
        else:
            if ctx is None:
                ctx = pa.cuda.Context()
            if not isinstance(ctx, pa.cuda.Context):
                ctx = pa.cuda.Context.from_numba(ctx)

            if cuda_data.dtype.char == 'M':
                cuda_data = cuda_data.view('uint64')
            cuda_data = ctx.buffer_from_data(cuda_data, size=self.alloc_size)

        self.cuda_data = cuda_data  # CudaBuffer

    def __repr__(self):
        return (f'{type(self).__name__}({self.shape}, strides={self.strides},'
                f'dtype={self.dtype}, cuda_data={self.cuda_data})')

    @classmethod
    def fromarray(cls, arr, ctx=None):
        if isinstance(arr, cls):
            return arr
        arr = np.asarray(arr)
        # TODO: order='C'?
        return cls(arr.shape, strides=arr.strides,
                   dtype=arr.dtype, cuda_data=arr, ctx=ctx)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        if axes:
            if tuple(axes) == tuple(range(self.ndim)):
                return self
        elif self.ndim <= 1:
            return self
        raise NotImplementedError('transpoose')

    # numba.cuda support

    __cuda_ndarray__ = True

    @property
    def gpu_data(self):
        return self.cuda_data.to_numba()

    def toDeviceNDArray(self):
        from numba.cuda.cudadrv.devicearray import DeviceNDArray
        return DeviceNDArray(self.shape, self.strides,
                             self.dtype, gpu_data=self.gpu_data)

    @classmethod
    def fromDeviceNDArray(cls, darr):
        cuda_data = pa.cuda.CudaBuffer.from_numba(darr.gpu_data)
        return cls(darr.shape, strines=darr.strides,
                   dtype=darr.dtype, cuda_data=cuda_data)

    __cuda_memory__ = True

    @property
    def device_pointer(self):
        if self.cuda_data is None:
            return c_void_p(0)
        return c_void_p(self.cuda_data.address)

    device_ctypes_pointer = device_pointer

    @property
    def __cuda_array_interface__(self):
        return {
            'shape': tuple(self.shape),
            'strides': tuple(self.strides),
            'data': (self.cuda_data.address or None, False),
            'typestr': self.dtype.str,
            'version': 0,
        }

    @property
    def _numba_type_(self):
        """
        Magic attribute expected by Numba to get the numba type that
        represents this object.
        """
        dtype = numpy_support.from_dtype(self.dtype)
        return types.Array(dtype, self.ndim, 'A')

    # eof numba.cuda support

    def __getitem__(self, item):
        d = self._dummy[item]
        return_element = d.shape == ()
        if return_element:  # element indexing
            if not isinstance(item, tuple):
                item = (item, )
            s = []
            for i in item:
                if i < 0:
                    s.append(slice(i, i-1, -1))
                else:
                    s.append(slice(i, i+1))
            d = self._dummy[tuple(s)]
        offset = d.get_offset()
        offset_last = d[tuple(slice(i-1, i) for i in d.shape)].get_offset()
        length = offset_last - offset + self.dtype.itemsize
        cuda_data = self.cuda_data.slice(offset=offset, length=length)
        if return_element:
            arr = np.frombuffer(cuda_data.copy_to_host(), dtype=self.dtype)
            assert len(arr) == 1, repr(arr.shape)
            return arr[0]
        return type(self)(d.shape, strides=d.strides,
                          dtype=d.dtype, cuda_data=cuda_data)

    def __setitem__(self, item, value):
        d = self._dummy[item]
        if d.shape == ():  # element indexing, converting to one element slice
            if not isinstance(item, tuple):
                item = (item, )
            s = []
            for i in item:
                if i < 0:
                    s.append(slice(i, i-1, -1))
                else:
                    s.append(slice(i, i+1))
            d = self._dummy[tuple(s)]
        offset = d.get_offset()
        length = d[tuple(slice(i-1, i) for i in d.shape)].get_offset()
        length += - offset + self.dtype.itemsize
        cuda_data = self.cuda_data.slice(offset=offset, length=length)

        # TODO: eliminate unnecessary copy to host when possible
        buf = cuda_data.copy_to_host()
        arr = np.frombuffer(buf, dtype=self.dtype)
        arr = np.lib.stride_tricks.as_strided(arr, shape=d.shape,
                                              strides=d.strides)
        arr[:] = value
        cuda_data.copy_from_host(buf)

    def copy_to_device(self, ary):
        if ary.size == 0:
            return
        if isinstance(ary, type(self)):
            nbytes = min(ary.alloc_size, self.alloc_size)
            self.cuda_data.copy_from_device(ary.cuda_data, nbytes=nbytes)
            return
        ary = np.array(
            ary,
            order='C' if self.flags.c_contiguous else 'F',
            subok=True,
            copy=False)
        nbytes = min(ary.nbytes, self.alloc_size)
        self.cuda_data.copy_from_host(ary, nbytes=nbytes)
        self.cuda_data.context.synchronize()

    def copy_to_host(self, ary=None):
        if ary is None:
            hostary = self.cuda_data.copy_to_host()
            hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                 strides=(self.strides
                                          if self.size != 0 else None),
                                 buffer=hostary)
        else:
            if ary.dtype != self.dtype:
                raise TypeError('incompatible dtype')
            hostary = ary
            if self.size > 0:
                buf = pa.py_buffer(hostary)
                self.cuda_data.copy_to_host(buf=buf)
        return hostary

    def split(self, section):
        raise NotImplementedError('split(section)')

    def get_ipc_handle(self):
        return self.cuda_data.export_for_ipc()

    def __array__(self, dtype=None):
        return self.copy_to_host().__array__(dtype)

    def __len__(self):
        return self.shape[0]

    def is_f_contiguous(self):
        return self._dummy.flags.f_contiguous

    def is_c_contiguous(self):
        return self._dummy.flags.c_contiguous

    @property
    def flags(self):
        return self._dummy.flags

    def reshape(self, *newshape, **kws):
        new = self._dummy.reshape(*newshape, **kws)
        new_order = 'C' if new.flags.c_contiguous else 'F'
        return type(self)(new.shape, strides=new.strides, dtype=new.dtype,
                          cuda_data=self.cuda_data, order=new_order)

    def ravel(self, order='C'):
        new = self._dummy.ravel(order=order)  # TODO: eliminate making a copy
        if self._dummy is not new.base:
            raise NotImplementedError('ravel requires a copy')
        new_order = 'C' if new.flags.c_contiguous else 'F'
        return type(self)(new.shape, strides=new.strides, dtype=new.dtype,
                          cuda_data=self.cuda_data, order=new_order)
