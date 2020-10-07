# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import Enum

import pandas as pd
import numba
import numpy as np
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector
from libcpp.utility cimport move
from cudf.utils import cudautils

from cudf._lib.types import np_to_cudf_types, cudf_to_np_types, NullHandling
from cudf._lib.types cimport (
    underlying_type_t_interpolation,
    underlying_type_t_null_policy,
    underlying_type_t_type_id,
)
from cudf._lib.types import Interpolation

try:
    # Numba >= 0.49
    from numba.np import numpy_support
except ImportError:
    # Numba <= 0.49
    from numba import numpy_support

cimport cudf._lib.cpp.types as libcudf_types
cimport cudf._lib.cpp.aggregation as libcudf_aggregation


class AggregationKind(Enum):
    SUM = libcudf_aggregation.aggregation.Kind.SUM
    PRODUCT = libcudf_aggregation.aggregation.Kind.PRODUCT
    MIN = libcudf_aggregation.aggregation.Kind.MIN
    MAX = libcudf_aggregation.aggregation.Kind.MAX
    COUNT = libcudf_aggregation.aggregation.Kind.COUNT_VALID
    SIZE = libcudf_aggregation.aggregation.Kind.COUNT_ALL
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
    NTH = libcudf_aggregation.aggregation.Kind.NTH_ELEMENT
    COLLECT = libcudf_aggregation.aggregation.Kind.COLLECT
    PTX = libcudf_aggregation.aggregation.Kind.PTX
    CUDA = libcudf_aggregation.aggregation.Kind.CUDA


cdef class Aggregation:

    def __init__(self, op, **kwargs):
        self.c_obj = move(make_aggregation(op, kwargs))

    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name.lower()


cdef unique_ptr[aggregation] make_aggregation(op, kwargs={}) except *:
    """
    Parameters
    ----------
    op : str or callable
        If callable, must meet one of the following requirements:

        * Is of the form lambda x: x.agg(*args, **kwargs), where
          `agg` is the name of a supported aggregation. Used to
          to specify aggregations that take arguments, e.g.,
          `lambda x: x.quantile(0.5)`.
        * Is a user defined aggregation function that operates on
          group values. In this case, the output dtype must be
          specified in the `kwargs` dictionary.

    Returns
    -------
    unique_ptr[aggregation]
    """
    cdef Aggregation agg
    if isinstance(op, str):
        agg = getattr(_AggregationFactory, op)(**kwargs)
    elif callable(op):
        if op is list:
            agg = _AggregationFactory.collect()
        elif "dtype" in kwargs:
            agg = _AggregationFactory.from_udf(op, **kwargs)
        else:
            agg = op(_AggregationFactory)
    else:
        raise TypeError("Unknown aggregation {}".format(op))
    return move(agg.c_obj)

# The Cython pattern below enables us to create an Aggregation
# without ever calling its `__init__` method, which would otherwise
# result in a RecursionError.
cdef class _AggregationFactory:

    @classmethod
    def sum(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_sum_aggregation())
        return agg

    @classmethod
    def min(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_min_aggregation())
        return agg

    @classmethod
    def max(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_max_aggregation())
        return agg

    @classmethod
    def mean(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_mean_aggregation())
        return agg

    @classmethod
    def count(cls, dropna=True):
        cdef libcudf_types.null_policy c_null_handling
        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_count_aggregation(
            c_null_handling
        ))
        return agg

    @classmethod
    def size(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_count_aggregation(
            <libcudf_types.null_policy><underlying_type_t_null_policy>(
                NullHandling.INCLUDE
            )
        ))
        return agg

    @classmethod
    def nunique(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_nunique_aggregation())
        return agg

    @classmethod
    def nth(cls, libcudf_types.size_type size):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(
            libcudf_aggregation.make_nth_element_aggregation(size)
        )
        return agg

    @classmethod
    def any(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_any_aggregation())
        return agg

    @classmethod
    def all(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_all_aggregation())
        return agg

    @classmethod
    def product(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_product_aggregation())
        return agg

    @classmethod
    def sum_of_squares(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_sum_of_squares_aggregation())
        return agg

    @classmethod
    def var(cls, ddof=1):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_variance_aggregation(ddof))
        return agg

    @classmethod
    def std(cls, ddof=1):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_std_aggregation(ddof))
        return agg

    @classmethod
    def median(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_median_aggregation())
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
        agg.c_obj = move(
            libcudf_aggregation.make_quantile_aggregation(c_q, c_interp)
        )
        return agg

    @classmethod
    def collect(cls):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)
        agg.c_obj = move(libcudf_aggregation.make_collect_aggregation())
        return agg

    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        cdef Aggregation agg = Aggregation.__new__(Aggregation)

        cdef libcudf_types.type_id tid
        cdef libcudf_types.data_type out_dtype
        cdef string cpp_str

        # Handling UDF type
        nb_type = numpy_support.from_dtype(kwargs['dtype'])
        type_signature = (nb_type[:],)
        compiled_op = cudautils.compile_udf(op, type_signature)
        output_np_dtype = np.dtype(compiled_op[1])
        cpp_str = compiled_op[0].encode('UTF-8')
        if output_np_dtype not in np_to_cudf_types:
            raise TypeError(
                "Result of window function has unsupported dtype {}"
                .format(op[1])
            )
        tid = (
            <libcudf_types.type_id> (
                <underlying_type_t_type_id> (
                    np_to_cudf_types[output_np_dtype]
                )
            )
        )
        out_dtype = libcudf_types.data_type(tid)

        agg.c_obj = move(libcudf_aggregation.make_udf_aggregation(
            libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
        ))
        return agg
