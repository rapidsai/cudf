import cffi

ffibuilder = cffi.FFI()
ffibuilder.set_source("libgdf_cffi.libgdf_cffi", None)

for fname in ['types.h', 'convert_types.h', 'functions.h', 'io_types.h', 'io_functions.h']:
    with open('include/gdf/cffi/{}'.format(fname), 'r') as fin:
        ffibuilder.cdef(fin.read())

if __name__ == "__main__":
    ffibuilder.compile()
