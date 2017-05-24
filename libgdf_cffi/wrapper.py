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
                        raw = self._api.gdf_error_get_name(errcode)
                        errname = self._ffi.string(raw).decode('ascii')
                        raise GDFError(errcode, errname)

                wrap.__name__ = fn.__name__
                self._cached[name] = wrap
            else:
                self._cached[name] = fn

            return self._cached[name]
