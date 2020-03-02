# Copyright (c) 2020, NVIDIA CORPORATION.

import pandas as pd
import numba
import numpy as np
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from cudf.utils import cudautils

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types
from cudf._libxx.move cimport move

from cudf._libxx.cpp.types cimport (
    type_id,
    size_type,
    data_type,
    interpolation
)
cimport cudf._libxx.cpp.aggregation as libcudf_aggregation


cdef class Aggregation:

    def __cinit__(self, op, *args, **kwargs):
        """
        op : func or str
        """
        cdef type_id tid
        cdef data_type out_dtype
        cdef string cpp_str

        if op == "sum":
            self.c_obj = move(libcudf_aggregation.make_sum_aggregation())
        elif op == "min":
            self.c_obj = move(libcudf_aggregation.make_min_aggregation())
        elif op == "max":
            self.c_obj = move(libcudf_aggregation.make_max_aggregation())
        elif op == "mean":
            self.c_obj = move(libcudf_aggregation.make_mean_aggregation())
        elif op == "count":
            self.c_obj = move(libcudf_aggregation.make_count_aggregation())
        elif op == "nunique":
            self.c_obj = move(libcudf_aggregation.make_nunique_aggregation())
        elif callable(op):
            # Handling UDF type
            nb_type = numba.numpy_support.from_dtype(kwargs['dtype'])
            type_signature = (nb_type[:],)
            compiled_op = cudautils.compile_udf(op, type_signature)
            output_np_dtype = np.dtype(compiled_op[1])
            cpp_str = compiled_op[0].encode('UTF-8')
            if output_np_dtype not in np_to_cudf_types:
                raise TypeError(
                    "Result of window function has unsupported dtype {}"
                    .format(op[1])
                )
            tid = np_to_cudf_types[output_np_dtype]

            out_dtype = data_type(tid)

            self.c_obj = move(libcudf_aggregation.make_udf_aggregation(
                libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
            ))
        else:
            raise TypeError("Invalid aggreagtion operation")
