"""
Uses radixsort implementation from CUB which has the following license:

Copyright (c) 2011, Duane Merrill.  All rights reserved.
Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
   Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
   Neither the name of the NVIDIA CORPORATION nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import print_function, absolute_import, division
import ctypes
from .common import load_lib
from contextlib import contextmanager
from numba.cuda.cudadrv.driver import device_pointer
from numba.cuda.cudadrv.drvapi import cu_stream
from numba.cuda.cudadrv.devicearray import auto_device, is_cuda_ndarray
from numba import cuda
import numpy as np

lib = load_lib('radixsort')

_argtypes = [
    ctypes.c_void_p, # temp
    ctypes.c_uint, # count
    ctypes.c_void_p, # d_key
    ctypes.c_void_p, # d_key_alt
    ctypes.c_void_p, # d_vals
    ctypes.c_void_p, # d_vals_alt
    cu_stream,
    ctypes.c_int, # descending
    ctypes.c_uint, # begin_bit
    ctypes.c_uint, # end_bit
]

_support_types = {
    np.float32: 'float',
    np.float64: 'double',
    np.int32: 'int32',
    np.uint32: 'uint32',
    np.int64: 'int64',
    np.uint64: 'uint64'
}

_overloads = {}


def _init():
    for ty, name in _support_types.items():
        dtype = np.dtype(ty)
        fn = getattr(lib, "radixsort_{0}".format(name))
        _overloads[dtype] = fn
        fn.argtypes = _argtypes
        fn.restype = ctypes.c_void_p


_init()

lib.radixsort_cleanup.argtypes = [ctypes.c_void_p]


def _devptr(p):
    if p is None:
        return None
    else:
        return device_pointer(p)


@contextmanager
def _autodevice(ary, stream, firstk=None):
    if ary is not None:
        dptr, conv = auto_device(ary, stream=stream)
        yield dptr
        if conv:
            if firstk is None:
                dptr.copy_to_host(ary, stream=stream)
            else:
                dptr.bind(stream)[:firstk].copy_to_host(ary[:firstk],
                                                        stream=stream)
    else:
        yield None


@cuda.jit
def _cu_arange(ary, count):
    i = cuda.grid(1)
    if i < count:
        ary[i] = i


class RadixSort(object):
    """Provides radix sort and radix select.

    The algorithm implemented here is best for large arrays (``N > 1e6``) due to
    the latency introduced by its use of multiple kernel launches. It is
    recommended to use ``segmented_sort`` instead for batches of smaller arrays.

    :type maxcount: int
    :param maxcount: Maximum number of items to sort
    :type dtype: numpy.dtype
    :param dtype: The element type to sort
    :type descending: bool
    :param descending: Sort in descending order?
    :param stream: The CUDA stream to run the kernels in
    """

    def __init__(self, maxcount, dtype, descending=False, stream=0):
        self.maxcount = int(maxcount)
        self.dtype = np.dtype(dtype)
        self._arysize = int(self.maxcount * self.dtype.itemsize)
        self.descending = descending
        self.stream = stream
        self._sort = _overloads[self.dtype]
        self._cleanup = lib.radixsort_cleanup

        ctx = cuda.current_context()
        self._temp_keys = ctx.memalloc(self._arysize)
        self._temp_vals = ctx.memalloc(self._arysize)
        self._temp = self._call(temp=None, keys=None, vals=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def close(self):
        """Explicitly release internal resources

        Called automatically when the object is deleted.
        """
        if self._temp is not None:
            self._cleanup(self._temp)
            self._temp = None

    def _call(self, temp, keys, vals, begin_bit=0, end_bit=None):
        stream = self.stream.handle if self.stream else self.stream
        begin_bit = begin_bit
        end_bit = end_bit or self.dtype.itemsize * 8
        descending = int(self.descending)

        count = self.maxcount
        if keys:
            count = keys.size

        return self._sort(
            temp,
            ctypes.c_uint(count),
            _devptr(keys),
            _devptr(self._temp_keys),
            _devptr(vals),
            _devptr(self._temp_vals),
            stream,
            descending,
            begin_bit,
            end_bit
        )

    def _sentry(self, ary):
        if ary.dtype != self.dtype:
            raise TypeError("dtype mismatch")
        if ary.size > self.maxcount:
            raise ValueError("keys array too long")

    def sort(self, keys, vals=None, begin_bit=0, end_bit=None):
        """
        Perform a inplace sort on ``keys``.  Memory transfer is performed
        automatically.

        :type keys: numpy.ndarray
        :param keys: Keys to sort inplace
        :type vals: numpy.ndarray
        :param vals: Optional. Additional values to be reordered along the sort.
                     It is modified in place. Only the ``uint32`` dtype is
                     supported in this version.
        :type begin_bit: int
        :param begin_bit: The first bit to sort
        :type end_bit: int
        :param end_bit: Optional. The last bit to sort
        """
        self._sentry(keys)
        with _autodevice(keys, self.stream) as d_keys:
            with _autodevice(vals, self.stream) as d_vals:
                self._call(self._temp, keys=d_keys, vals=d_vals,
                           begin_bit=begin_bit, end_bit=end_bit)

    def select(self, k, keys, vals=None, begin_bit=0, end_bit=None):
        """Perform a inplace k-select on ``keys``.

        Memory transfer is performed automatically.

        :type keys: numpy.ndarray
        :param keys: Keys to sort inplace
        :type vals: numpy.ndarray
        :param vals: Optional. Additional values to be reordered along the sort.
                     It is modified in place. Only the ``uint32`` dtype is
                     supported in this version.
        :type begin_bit: int
        :param begin_bit: The first bit to sort
        :type end_bit: int
        :param end_bit: Optional. The last bit to sort
        """
        self._sentry(keys)
        with _autodevice(keys, self.stream, firstk=k) as d_keys:
            with _autodevice(vals, self.stream, firstk=k) as d_vals:
                self._call(self._temp, keys=d_keys, vals=d_vals,
                           begin_bit=begin_bit, end_bit=end_bit)

    def init_arg(self, size):
        """Initialize an empty CUDA ndarray of uint32 with ascending integers
        starting from zero

        :type size: int
        :param size: Number of elements for the output array
        :return: An array with values ``[0, 1, 2, ...m size - 1 ]``
        """
        d_vals = cuda.device_array(size, dtype=np.uint32, stream=self.stream)
        _cu_arange.forall(d_vals.size, stream=self.stream)(d_vals, size)
        return d_vals

    def argselect(self, k, keys, begin_bit=0, end_bit=None):
        """Similar to ``RadixSort.select`` but returns the new sorted indices.

        :type keys: numpy.ndarray
        :param keys: Keys to sort inplace
        :type begin_bit: int
        :param begin_bit: The first bit to sort
        :type end_bit: int
        :param end_bit: Optional. The last bit to sort
        :return: The indices indicating the new ordering as an array on the CUDA
                 device or on the host.
        """
        d_vals = self.init_arg(keys.size)
        self.select(k, keys, vals=d_vals, begin_bit=begin_bit, end_bit=end_bit)
        res = d_vals.bind(self.stream)[:k]
        if not is_cuda_ndarray(keys):
            res = res.copy_to_host(stream=self.stream)
        return res

    def argsort(self, keys, begin_bit=0, end_bit=None):
        """Similar to ``RadixSort.sort`` but returns the new sorted indices.

        :type keys: numpy.ndarray
        :param keys: Keys to sort inplace
        :type begin_bit: int
        :param begin_bit: The first bit to sort
        :type end_bit: int
        :param end_bit: Optional. The last bit to sort
        :return: The indices indicating the new ordering as an array on the CUDA
                 device or on the host.
        """
        d_vals = self.init_arg(keys.size)
        self.sort(keys, vals=d_vals, begin_bit=begin_bit, end_bit=end_bit)
        res = d_vals
        if not is_cuda_ndarray(keys):
            res = res.copy_to_host(stream=self.stream)
        return res

