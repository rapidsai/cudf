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

from cudf._libxx.aggregation cimport get_aggregation
from cudf._libxx.cpp.types cimport (
    type_id,
    size_type,
    data_type,
    interpolation
)
cimport cudf._libxx.cpp.aggregation as libcudf_aggregation


# need to update as and when we add new aggregations with additional options
cdef unique_ptr[aggregation] get_aggregation(op, kwargs) except *:

    cdef type_id tid
    cdef data_type out_dtype
    cdef string cpp_str
    cdef unique_ptr[aggregation] agg

    if op == "sum":
        agg = move(libcudf_aggregation.make_sum_aggregation())
    elif op == "min":
        agg = move(libcudf_aggregation.make_min_aggregation())
    elif op == "max":
        agg = move(libcudf_aggregation.make_max_aggregation())
    elif op == "mean":
        agg = move(libcudf_aggregation.make_mean_aggregation())
    elif op == "count":
        agg = move(libcudf_aggregation.make_count_aggregation())
    elif op == "any":
        agg = move(libcudf_aggregation.make_any_aggregation())
    elif op == "all":
        agg = move(libcudf_aggregation.make_all_aggregation())
    elif op == "product":
        agg = move(libcudf_aggregation.make_product_aggregation())
    elif op == "sum_of_squares":
        agg = move(libcudf_aggregation.make_sum_of_squares_aggregation())
    elif op == "var":
        agg = move(libcudf_aggregation.make_variance_aggregation(
            kwargs['ddof'])
        )
    elif op == "std":
        agg = move(libcudf_aggregation.make_std_aggregation(kwargs['ddof']))
    elif op == "count":
        agg = move(libcudf_aggregation.make_count_aggregation())
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

        agg = move(libcudf_aggregation.make_udf_aggregation(
            libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
        ))
    else:
        assert False, "Invalid aggreagtion operation"

    return move(agg)
