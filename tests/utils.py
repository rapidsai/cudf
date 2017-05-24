from libgdf_cffi import ffi, libgdf


def new_column():
    return ffi.new('gdf_column*')


def unwrap_devary(devary):
    return ffi.cast('void*', devary.device_ctypes_pointer.value)

