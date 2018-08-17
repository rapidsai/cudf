import cffi

ffibuilder = cffi.FFI()
ffibuilder.set_source("librmm_cffi.librmm_cffi", None)

with open('include/memory.h', 'r') as fin:
    ffibuilder.cdef(fin.read())

if __name__ == "__main__":
    ffibuilder.compile()
