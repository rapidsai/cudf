import os
from cffi

ffibuilder = cffi.FFI()

PATH = os.path.dirname(os.path.realpath("__file__"))

ffibuilder.set_source("librmm_cffi.librmm_cffi", None)

with open(os.path.join(PATH, "memory.c"), "r") as fin:
  ffibuilder.cdef(fin.read())

if __name__ == "__main__":
  ffibuilder.compile()