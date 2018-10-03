"""Defines DeviceNDArray with a similar interface to
numba.cuda.DeviceNDArray.
"""
# Author: Pearu Peterson
# Created: September 2018

from ctypes import c_void_p
import numpy as np
import pyarrow as pa
import pyarrow.cuda
from . import dummyarray

integer_types = (int,)


def from_array_like(ary, gpu_data=None):
    "Create a DeviceNDArray object that is like ary."
    if ary.ndim == 0:
        ary = ary.reshape(1)
    return CudaNDArray(ary.shape, ary.strides, ary.dtype, gpu_data=gpu_data)


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
                raise ValueError('order must be "C" or "F" but got %r' % (order))
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
    
    #__cuda_memory__ = True # numba.cuda support

    def __init__(self, shape, strides=None, dtype=None, cuda_data=None,
                 order='C', ctx = None):
        self.shape, self.strides, self.dtype \
            = prepare_shape_strides_dtype(shape, strides, dtype, order=order)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.alloc_size = self.size * self.dtype.itemsize
        self._dummy = dummyarray.Array(self.shape, strides=self.strides, dtype=self.dtype, order=order)
        if cuda_data is None:
            if ctx is None:
                ctx = pa.cuda.Context()
            cuda_data = ctx.new_buffer(self.size)
        elif isinstance(cuda_data, pa.cuda.CudaBuffer):
            pass
        elif isinstance(cuda_data, CudaNDArray):
            raise NotImplementedError('CUDA device array from a device array')
        else:
            if ctx is None:
                ctx = pa.cuda.Context()
            cuda_data = ctx.buffer_from_data(cuda_data, size=self.alloc_size)

        self.cuda_data = cuda_data # CudaBuffer

    def __repr__(self):
        return type(self).__name__ + '({shape}, strides={strides}, dtype={dtype}, cuda_data={cuda_data})'.format_map(self.__dict__)
        
    @classmethod
    def fromarray(cls, arr):
        arr = np.asarray(arr)
        return cls(arr.shape, strides=arr.strides, dtype=arr.dtype, cuda_data = arr)

    @property
    def gpu_data(self):
        from numba.cuda.cudadrv.driver import MemoryPointer
        context = self.cuda_data.context # TODO: this is pyarrow context, MemoryPointer might require numba.cuda context
        pointer = c_void_p(self.cuda_data.address)
        size = self.cuda_data.size
        return MemoryPointer(context, pointer=pointer, size=size)
    
    def toDeviceNDArray(self):
        from numba.cuda.cudadrv.devicearray import DeviceNDArray
        return DeviceNDArray(self.shape, self.strides, self.dtype, gpu_data = self.gpu_data)

    @classmethod
    def fromDeviceNDArray(cls, darr): # NOT FUNCTIONAL
        address = darr.gpu_data.device_pointer.value
        size = darr.size
        context = darr.context # TODO: check if correct
        cuda_data = pa.cuda.CudaBuffer(address, size, context) # not supported currently in pyarrow.cuda
        return cls(darr.shape, strines=darr.strides ,dtype=darr.dtype, cuda_data=cuda_data)
        
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

    @property
    def device_ctypes_pointer(self):  # numba.cuda support
        if self.cuda_data is None:
            return c_void_p(0)
        return c_void_p(self.cuda_data.address)

    @property
    def __cuda_array_interface__(self):  # numba.cuda support
        return {
            'shape': tuple(self.shape),
            'strides': tuple(self.strides),
            'data': (self.device_ctypes_pointer.value, False),
            'typestr': self.dtype.str,
            'version': 0,
        }

    def __getitem__(self, item):
        d = self._dummy[item]
        return_element = d.shape==()
        if return_element: # element indexing
            if not isinstance(item, tuple):
                item = (item,)
            s = []
            for i in item:
                if i<0:
                    s.append(slice(i,i-1,-1))
                else:
                    s.append(slice(i,i+1))
            d = self._dummy[tuple(s)]
        offset = d.get_offset()
        length = d[tuple(slice(i-1,i) for i in d.shape)].get_offset() - offset + self.dtype.itemsize
        cuda_data = self.cuda_data.slice(offset=offset, length=length)
        if return_element:
            arr = np.frombuffer(cuda_data.copy_to_host(), dtype=self.dtype)
            assert len(arr)==1,repr(arr.shape)
            return arr[0]
        return type(self)(d.shape, strides=d.strides, dtype=d.dtype, cuda_data=cuda_data)

    def __setitem__(self, item, value):
        d = self._dummy[item]
        if d.shape==(): # element indexing, converting to one element slice
            if not isinstance(item, tuple):
                item = (item,)
            s = []
            for i in item:
                if i<0:
                    s.append(slice(i,i-1,-1))
                else:
                    s.append(slice(i,i+1))
            d = self._dummy[tuple(s)]
        offset = d.get_offset()
        length = d[tuple(slice(i-1,i) for i in d.shape)].get_offset() - offset + self.dtype.itemsize
        cuda_data = self.cuda_data.slice(offset=offset, length=length)

        # TODO: eliminate unnecessary copy to host when possible
        buf = cuda_data.copy_to_host()
        arr = np.frombuffer(buf, dtype=self.dtype)
        arr = np.lib.stride_tricks.as_strided(arr, shape=d.shape, strides=d.strides)
        arr[:] = value
        cuda_data.copy_from_host(buf)
        
    
    def copy_to_device(self, ary):
        if ary.size == 0:
            return
        ary = np.array(
            ary,
            order='C' if self.flags.c_contiguous else 'F',
            subok=True,
            copy=False)
        self.cuda_data.copy_from_host(ary, nbytes = min(ary.size, self.size))

    def copy_to_host(self, ary=None):
        if ary is None:
            hostary = self.cuda_data.copy_to_host()
            hostary = np.ndarray(shape=self.shape, dtype=self.dtype,
                                 strides=(self.strides if self.size!=0 else None),
                                 buffer=hostary)
        else:
            if ary.dtype != self.dtype:
                raise TypeError('incompatible dtype')
            hostary = ary
            if self.size > 0:
                buf = pa.py_buffer(hostary)
                self.cuda_data.copy_to_host(buf = buf)
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
        return type(self)(new.shape, strides=new.strides, dtype=new.dtype, cuda_data=self.cuda_data, order=new_order)

    def ravel(self, order='C'):
        new = self._dummy.ravel(order=order) # TODO: eliminate making a copy
        if self._dummy is not new.base:
            raise NotImplementedError('ravel requires a copy')
        new_order = 'C' if new.flags.c_contiguous else 'F'
        return type(self)(new.shape, strides=new.strides, dtype=new.dtype, cuda_data=self.cuda_data, order=new_order)
