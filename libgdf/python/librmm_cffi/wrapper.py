# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numba import cuda
import ctypes
from librmm_cffi import librmm_config as rmm_cfg

class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


class _RMMWrapper(object):
    def __init__(self, ffi, api):
        self._ffi = ffi
        self._api = api
        self._cached = {}

    def __getattr__(self, name):
        try:
            return self._cached[name]
        except KeyError:
            fn = getattr(self._api, name)

            # hack to check the return type
            textrepr = str(fn)
            if 'gdf_error(*)' in textrepr:
                def wrap(*args):
                    # convert errcode to exception
                    errcode = fn(*args)
                    if errcode != self._api.RMM_SUCCESS:
                        errname, msg = self._get_error_msg(errcode)
                        raise RMMError(errname, msg)

                wrap.__name__ = fn.__name__
                self._cached[name] = wrap
            else:
                self._cached[name] = fn

            return self._cached[name]

    def _ffi_str(self, strptr):
        """Convert CFFI const char * into a str.
        """
        return self._ffi.string(strptr).decode('ascii')

    def _get_error_msg(self, errcode):
        """Get error message for the given error code.
        """
        if errcode == self._api.RMM_CUDA_ERROR:
            cudaerr = self._api.gdf_cuda_last_error()
            errname = self._ffi_str(self._api.gdf_cuda_error_name(cudaerr))
            details = self._ffi_str(self._api.gdf_cuda_error_string(cudaerr))
            msg = 'CUDA ERROR. {}: {}'.format(errname, details)
        else:
            errname = self._ffi_str(self._api.gdf_error_get_name(errcode))
            msg = errname
        return errname, msg

    def initialize(self):
        """Initializes the RMM library using the options set in the 
           librmm_config module
        """
        opts = self._ffi.new("rmmOptions_t *",
                             [rmm_cfg.use_pool_allocator, 
                              rmm_cfg.initial_pool_size, 
                              rmm_cfg.enable_logging])
        return self.rmmInitialize(opts)

    def finalize(self):
        """Finalizes the RMM library, freeing all allocated memory
        """
        return self.rmmFinalize()

    def csv_log(self):
        """Returns a CSV log of all events logged by RMM, if logging is 
           enabled
        """
        logsize = self.rmmLogSize()
        buf = self._ffi.new("char[]", logsize)
        self.rmmGetLog(buf, logsize)
        return self._ffi.string(buf).decode('utf-8')

    def _array_helper(self, addr, datasize, shape,
                      strides, dtype, finalizer=None):
        ctx = cuda.current_context()
        ptr = ctypes.c_uint64(int(addr))
        mem = cuda.driver.MemoryPointer(ctx, ptr, datasize,
                                        finalizer=finalizer)
        return cuda.cudadrv.devicearray.DeviceNDArray(shape, strides, dtype,
                                                      gpu_data=mem)

    def device_array_from_ptr(self, ptr, nelem,
                              dtype=np.float, finalizer=None):
        """device_array_from_ptr(self, ptr, size, dtype=np.float, stream=0)

        Create a Numba device array from a ptr, size, and dtype.
        """
        # Handle Datetime Column
        if dtype == np.datetime64:
            dtype = np.dtype('datetime64[ms]')
        else:
            dtype = np.dtype(dtype)

        elemsize = dtype.itemsize
        datasize = elemsize * nelem
        addr = self._ffi.cast("uintptr_t", ptr)
        # note no finalizer -- freed externally!
        return self._array_helper(addr=addr, datasize=datasize,
                                  shape=(nelem,), strides=(elemsize,),
                                  dtype=dtype, finalizer=finalizer)

    def device_array(self, shape, dtype=np.float, strides=None, order='C',
                     stream=0):
        """device_array(self, shape, dtype=np.float, strides=None, order='C',
                        stream=0)

        Allocate an empty Numba device array. Clone of Numba
        `cuda.device_array`, but uses RMM for device memory management.
        """
        shape, strides, dtype = cuda.api._prepare_shape_strides_dtype(shape,
                                                                      strides,
                                                                      dtype,
                                                                      order)
        datasize = cuda.driver.memory_size_from_info(shape, strides,
                                                     dtype.itemsize)

        ptr = self._ffi.new("void **")
        self._api.rmmAlloc(ptr, datasize,
                           self._ffi.cast("cudaStream_t", stream))
        addr = self._ffi.cast("uintptr_t*", ptr)[0]
        # Note Numba will call the finalizer to free the device memory
        # allocated above
        return self._array_helper(addr=addr, datasize=datasize,
                                  shape=shape, strides=strides, dtype=dtype,
                                  finalizer=self._make_finalizer(addr, stream))

    def device_array_like(self, ary, stream=0):
        """device_array_like(self, ary, stream=0)

        Call rmmlib.device_array with information from `ary`. Clone of Numba
        `cuda.device_array_like`, but uses RMM for device memory management.
        """
        if ary.ndim == 0:
            ary = ary.reshape(1)

        return self.device_array(ary.shape, ary.dtype, ary.strides,
                                 stream=stream)

    def to_device(self, ary, stream=0, copy=True, to=None):
        """to_device(self, ary, stream=0, copy=True, to=None)

        Allocate and transfer a numpy ndarray or structured scalar to the
        device. Clone of Numba `cuda.to_device`, but uses RMM for device
        memory management.
        """
        if to is None:
            to = self.device_array_like(ary, stream=stream)
            to.copy_to_device(ary, stream=stream)
            return to
        if copy:
            to.copy_to_device(ary, stream=stream)
        return to

    def auto_device(self, obj, stream=0, copy=True):
        """
        Create a DeviceRecord or DeviceArray like obj and optionally copy data
        from host to device. If obj already represents device memory, it is
        returned and no copy is made. Uses RMM for device memory allocation if
        necessary.
        """
        if cuda.driver.is_device_memory(obj):
            return obj, False
        else:
            if isinstance(obj, np.void):
                # raise NotImplementedError("DeviceRecord type not supported "
                #                            "by RMM")
                devobj = cuda.devicearray.from_record_like(obj, stream=stream)
            else:
                if not isinstance(obj, np.ndarray):
                    obj = np.asarray(obj)
                cuda.devicearray.sentry_contiguous(obj)
                devobj = self.device_array_like(obj, stream=stream)

            if copy:
                devobj.copy_to_device(obj, stream=stream)
            return devobj, True

    def get_ipc_handle(self, ary, stream=0):
        """
        Get an IPC handle from the DeviceArray ary with offset modified by
        the RMM memory pool.
        """
        ipch = cuda.devices.get_context().get_ipc_handle(ary.gpu_data)
        offset = self._ffi.new("offset_t*")
        ptr = self._ffi.cast("void*", ary.device_ctypes_pointer.value)
        self._api.rmmGetAllocationOffset(offset, ptr,
                                         self._ffi.cast("cudaStream_t",
                                                        stream))
        # replace offset with RMM's offset
        ipch.offset = offset[0]
        desc = dict(shape=ary.shape,
                    strides=ary.strides,
                    dtype=ary.dtype)
        return cuda.cudadrv.devicearray.IpcArrayHandle(ipc_handle=ipch,
                                                       array_desc=desc)

    def _make_finalizer(self, handle, stream):
        """Factory to make the finalizer function.
        We need to bind *handle* and *stream* into the actual finalizer,
        which takes no arg
        """
        def finalizer():
            """Invoked when the MemoryPointer is freed
            """
            cptr = self._ffi.cast("void*", handle)
            return self._api.rmmFree(cptr, self._ffi.cast("cudaStream_t",
                                                            stream))
        return finalizer
