class GDFError(Exception):
    def __init__(self, errcode, msg):
        self.errcode = errcode
        super(GDFError, self).__init__(msg)


class _libgdf_wrapper(object):
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
                    if errcode != self._api.GDF_SUCCESS:
                        errname, msg = self._get_error_msg(errcode)
                        raise GDFError(errname, msg)

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
        if errcode == self._api.GDF_CUDA_ERROR:
            cudaerr = self._api.gdf_cuda_last_error()
            errname = self._ffi_str(self._api.gdf_cuda_error_name(cudaerr))
            details = self._ffi_str(self._api.gdf_cuda_error_string(cudaerr))
            msg = 'CUDA ERROR. {}: {}'.format(errname, details)
        else:
            errname = self._ffi_str(self._api.gdf_error_get_name(errcode))
            msg = errname
        return errname, msg
