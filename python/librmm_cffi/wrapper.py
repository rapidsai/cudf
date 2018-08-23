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

    def device_array_from_ptr(self, ptr, size, dtype=np.float):
        """device_array_from_ptr(self, ptr, size, dtype=np.float, stream=0)

        Create a Numba device array from a ptr, size, and dtype.
        """
        shape, strides, dtype = cuda.api._prepare_shape_strides_dtype(size, (), dtype, 'C')
        bytesize = cuda.driver.memory_size_from_info(shape, strides, dtype.itemsize)

        ctx = cuda.current_context()
        ptr = cuda.driver.drvapi.cu_device_ptr(self._ffi.cast("uintptr_t", ptr))
        mem = cuda.driver.MemoryPointer(ctx, ptr, bytesize) # note no finalizer -- freed externally!
        return cuda.cudadrv.devicearray.DeviceNDArray(shape, strides, dtype, gpu_data=mem)

    def device_array(self, shape, dtype=np.float, strides=None, order='C', stream=0):
        """device_array(self, shape, dtype=np.float, strides=None, order='C', stream=0)

        Allocate an empty Numba device array. Clone of Numba `cuda.device_array`, but 
        uses RMM for device memory management.
        """
        shape, strides, dtype = cuda.api._prepare_shape_strides_dtype(shape, strides, dtype, order)
        bytesize = cuda.driver.memory_size_from_info(shape, strides, dtype.itemsize)


        cptr = self._ffi.new("void **")
        self._api.rmmAlloc(cptr, bytesize, self._ffi.cast("cudaStream_t", stream))

        ctx = cuda.current_context()
        ptr = cuda.driver.drvapi.cu_device_ptr(self._ffi.cast("uintptr_t*", cptr)[0])
        mem = cuda.driver.MemoryPointer(ctx, ptr, bytesize, finalizer=self._make_finalizer(ptr, stream))
        return cuda.cudadrv.devicearray.DeviceNDArray(shape, strides, dtype, gpu_data=mem)


    def device_array_like(self, ary, stream=0):
        """device_array_like(self, ary, stream=0)

        Call rmmlib.device_array with information from `ary`. Clone of Numba `cuda.device_array_like`,
        but uses RMM for device memory management.
        """
        if isinstance(ary, np.ndarray):
            size = ary.nbytes
        else:
            size = ary.gpu_data.size

        return self.device_array(ary.shape, ary.dtype, ary.strides, stream=stream)

    def to_device(self, ary, stream=0, copy=True, to=None):
        """to_device(self, ary, stream=0, copy=True, to=None)

        Allocate and transfer a numpy ndarray or structured scalar to the device.

        Clone of Numba `cuda.to_device`, but uses RMM for device memory management.
        """
        if to is None:
            to = self.device_array_like(ary, stream=stream)
            to.copy_to_device(ary, stream=stream)
            return to
        if copy:
            to.copy_to_device(ary, stream=stream)
        return to

    def _prepare_shape_strides_dtype(self, shape, strides, dtype, order):
        dtype = np.dtype(dtype)
        if isinstance(shape, (int, long)):
            shape = (shape,)
        if isinstance(strides, (int, long)):
            strides = (strides,)
        else:
            if shape == ():
                shape = (1,)
            strides = strides or self._fill_stride_by_order(shape, dtype, order)
        return shape, strides, dtype


    def _fill_stride_by_order(self, shape, dtype, order):
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
            raise ValueError('must be either C/F order')
        return tuple(strides)

    def _make_finalizer(self, handle, stream):
            """Factory to make the finalizer function.
            We need to bind *handle* and *stream* into the actual finalizer,
            which takes no arg
            """
            def finalizer():
                """Invoked when the MemoryPointer is freed
                """
                cptr = self._ffi.cast("void*", handle.value)
                return self._api.rmmFree(cptr, self._ffi.cast("cudaStream_t", stream))
            return finalizer