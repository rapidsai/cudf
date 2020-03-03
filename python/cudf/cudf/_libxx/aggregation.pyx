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


cdef unique_ptr[aggregation] make_aggregation(op, kwargs={}):
    cdef _Aggregation agg
    if isinstance(op, str):
        agg = getattr(_Aggregation, op)()
    elif callable(op):
        if "dtype" in kwargs:
            agg = _Aggregation.from_udf(op, **kwargs)
        else:
            agg = op(_Aggregation)
    return move(agg.c_obj)


cdef class _Aggregation:

    @classmethod
    def sum(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_sum_aggregation())
        return agg

    @classmethod
    def min(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_min_aggregation())
        return agg

    @classmethod
    def max(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_max_aggregation())
        return agg

    @classmethod
    def mean(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_mean_aggregation())
        return agg

    @classmethod
    def count(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_count_aggregation())
        return agg

    @classmethod
    def nunique(cls):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_nunique_aggregation())
        return agg

    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        cdef _Aggregation agg = _Aggregation.__new__(_Aggregation)

        cdef type_id tid
        cdef data_type out_dtype
        cdef string cpp_str

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

        agg.c_obj = move(libcudf_aggregation.make_udf_aggregation(
            libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
        ))
        return agg
