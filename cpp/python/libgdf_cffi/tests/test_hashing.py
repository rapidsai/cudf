#
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
#

from __future__ import print_function
from contextlib import contextmanager

import numpy as np

from libgdf_cffi import ffi, libgdf
from librmm_cffi import librmm as rmm

from libgdf_cffi.tests.utils import new_column, unwrap_devary, get_dtype


@contextmanager
def _make_hash_input(hash_input, ncols):
    ci = []
    di = []
    for i in range(ncols):
        di.append(rmm.to_device(hash_input[i]))

    for i in range(ncols):
        col_input = new_column()
        libgdf.gdf_column_view(col_input, unwrap_devary(di[i]), ffi.NULL,
                               hash_input[i].size,
                               get_dtype(hash_input[i].dtype))
        ci.append(col_input)

    initial_hash_values = rmm.to_device(np.arange(ncols, dtype=np.uint32))

    yield ci, unwrap_devary(initial_hash_values)


def _call_hash_multi(api, ncols, col_input, magic, initial_hash_values, nrows):
    out_ary = np.zeros(nrows, dtype=np.int32)
    d_out = rmm.to_device(out_ary)
    col_out = new_column()
    libgdf.gdf_column_view(col_out, unwrap_devary(d_out), ffi.NULL,
                           out_ary.size, get_dtype(d_out.dtype))

    api(ncols, col_input, magic, initial_hash_values, col_out)

    hashed_result = d_out.copy_to_host()
    print(hashed_result)

    return hashed_result


def test_hashing():
    # Make data
    nrows = 8
    # dtypes as number of columns
    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]

    hash_input = []
    for dt in dtypes:
        if(dt == np.float32):
            hi = np.array(np.random.randint(np.iinfo(np.int32).min,
                                            np.iinfo(np.int32).max,
                                            nrows),
                          dtype=dt)
        elif(dt == np.float64):
            hi = np.array(np.random.randint(np.iinfo(np.int64).min,
                                            np.iinfo(np.int64).max,
                                            nrows),
                          dtype=dt)
        else:
            hi = np.array(np.random.randint(np.iinfo(dt).min,
                                            np.iinfo(dt).max,
                                            nrows),
                          dtype=dt)
        hi[-1] = hi[0]
        hash_input.append(hi)

    ncols = len(hash_input)
    magic = libgdf.GDF_HASH_MURMUR3

    with _make_hash_input(hash_input, ncols) as (col_input, init_hash_values):
        # Hash
        for init_vals in [ffi.NULL, init_hash_values]:
            hashed_column = _call_hash_multi(libgdf.gdf_hash, ncols,
                                             col_input, magic, init_vals,
                                             nrows)

            # Check if first and last row are equal
            assert tuple([hashed_column[0]]) == tuple([hashed_column[-1]])
