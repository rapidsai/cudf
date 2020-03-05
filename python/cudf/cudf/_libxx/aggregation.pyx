# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import Enum

import pandas as pd
import numba
import numpy as np
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from cudf.utils import cudautils

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types
from cudf._libxx.move cimport move

cimport cudf._libxx.cpp.types as libcudf_types
cimport cudf._libxx.cpp.aggregation as libcudf_aggregation
from cudf._libxx.types cimport (
    underlying_type_t_interpolation
)
from cudf._libxx.types import Interpolation


class AggregationKind(Enum):
    SUM = libcudf_aggregation.aggregation.Kind.SUM
    PRODUCT = libcudf_aggregation.aggregation.Kind.PRODUCT
    MIN = libcudf_aggregation.aggregation.Kind.MIN
    MAX = libcudf_aggregation.aggregation.Kind.MAX
    COUNT = libcudf_aggregation.aggregation.Kind.COUNT_VALID
    ANY = libcudf_aggregation.aggregation.Kind.ANY
    ALL = libcudf_aggregation.aggregation.Kind.ALL
    SUM_OF_SQUARES = libcudf_aggregation.aggregation.Kind.SUM_OF_SQUARES
    MEAN = libcudf_aggregation.aggregation.Kind.MEAN
    VARIANCE = libcudf_aggregation.aggregation.Kind.VARIANCE
    STD = libcudf_aggregation.aggregation.Kind.STD
    MEDIAN = libcudf_aggregation.aggregation.Kind.MEDIAN
    QUANTILE = libcudf_aggregation.aggregation.Kind.QUANTILE
    ARGMAX = libcudf_aggregation.aggregation.Kind.ARGMAX
    ARGMIN = libcudf_aggregation.aggregation.Kind.ARGMIN
    NUNIQUE = libcudf_aggregation.aggregation.Kind.NUNIQUE
    NTH_ELEMENT = libcudf_aggregation.aggregation.Kind.NTH_ELEMENT
    PTX = libcudf_aggregation.aggregation.Kind.PTX
    CUDA = libcudf_aggregation.aggregation.Kind.CUDA


cdef class Aggregation:

    def __init__(self, op, **kwargs):
        self.c_obj = move(make_aggregation(op, kwargs))

    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name.lower()


cdef unique_ptr[aggregation] make_aggregation(op, kwargs={}) except *:
    cdef Aggregation agg
    if isinstance(op, str):
        agg = getattr(_AggregationFactory, op)(**kwargs)
    elif callable(op):
        if "dtype" in kwargs:
            agg = _AggregationFactory.from_udf(op, **kwargs)
        else:
            agg = op(_AggregationFactory)
    return move(agg.c_obj)


cdef class _AggregationFactory:

    @classmethod
    def sum(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_sum_aggregation())
        return agg

    @classmethod
    def min(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_min_aggregation())
        return agg

    @classmethod
    def max(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_max_aggregation())
        return agg

    @classmethod
    def mean(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_mean_aggregation())
        return agg

    @classmethod
    def count(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_count_aggregation())
        return agg

    @classmethod
    def nunique(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_nunique_aggregation())
        return agg

    @classmethod
    def any(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_any_aggregation())
        return agg

    @classmethod
    def all(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_all_aggregation())
        return agg

    @classmethod
    def product(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_product_aggregation())
        return agg

    @classmethod
    def sum_of_squares(cls, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_sum_of_squares_aggregation())
        return agg

    @classmethod
    def var(cls, ddof, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_variance_aggregation(ddof))
        return agg

    @classmethod
    def std(cls, ddof, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_std_aggregation(ddof))
        return agg

    @classmethod
    def quantile(cls, q=0.5, interpolation="linear"):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)

        if not pd.api.types.is_list_like(q):
            q = [q]

        cdef vector[double] c_q = q
        cdef libcudf_types.interpolation c_interp = (
            <libcudf_types.interpolation> (
                <underlying_type_t_interpolation> (
                    Interpolation[interpolation.upper()]
                )
            )
        )
        agg.c_obj = move(libcudf_aggregation.make_quantile_aggregation(c_q, c_interp))
        return agg

    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)

        cdef libcudf_types.type_id tid
        cdef libcudf_types.data_type out_dtype
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

        out_dtype = libcudf_types.data_type(tid)

        agg.c_obj = move(libcudf_aggregation.make_udf_aggregation(
            libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
        ))
        return agg
