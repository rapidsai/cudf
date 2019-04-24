# Copyright (c) 2018-2019, NVIDIA CORPORATION.

import cffi
import os
import sys

fnames = [
    'types.h',
    'convert_types.h',
    'functions.h',
    'io_types.h',
    'io_functions.h'
]

cudf_include_dirs = [
    os.environ.get('CUDF_INCLUDE_DIR', '../../include/cudf/'),
    os.path.join(sys.prefix, 'include/cudf')
]

for cudf_include_dir in cudf_include_dirs:
    if all([
        os.path.isfile(os.path.join(cudf_include_dir, fname))
        for fname in fnames
    ]):
        include_dir = cudf_include_dir
        break

ffibuilder = cffi.FFI()
ffibuilder.set_source("libgdf_cffi.libgdf_cffi", None)

for fname in fnames:
    with open(os.path.join(include_dir, fname), 'r') as fin:
        ffibuilder.cdef(fin.read())

if __name__ == "__main__":
    ffibuilder.compile()
