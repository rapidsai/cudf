import numpy as np
from numba import cuda 

class RMMError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(RMMError, self).__init__(msg)


class _librmm_wrapper(object):
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
                    # covert errcode to exception
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
        return self.rmmInitialize()

    def finalize(self):
        return self.rmmFinalize()

    def device_array_like(self, ary, stream=0):
        if isinstance(ary, np.ndarray):
            size = ary.nbytes
        else:
            size = ary.gpu_data.size

        cptr = self._ffi.new("void **")
        self._api.rmmAlloc(cptr, size, self._ffi.cast("cudaStream_t", stream))

        ctx = cuda.current_context()
        ptr = cuda.driver.drvapi.cu_device_ptr(self._ffi.cast("size_t*", cptr)[0])
        mem = cuda.driver.MemoryPointer(ctx, ptr, size)
        d_ary = cuda.cudadrv.devicearray.DeviceNDArray(ary.shape, ary.strides, ary.dtype, gpu_data=mem)
        return d_ary


    def to_device(self, ary, stream=0):
        d_ary = self.device_array_like(ary, stream)
        d_ary.copy_to_device(ary, stream)
        return d_ary

    def free_device_array_memory(self, ary, stream=0):
        cptr = self._ffi.cast("void*", ary.gpu_data.device_pointer.value)
        return self._api.rmmFree(cptr, self._ffi.cast("cudaStream_t", stream))
