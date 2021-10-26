# Copyright (c) 2020, NVIDIA CORPORATION.

from enum import Enum

import numba
import numpy as np
import pandas as pd

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move
from libcpp.vector cimport vector

from cudf._lib.types import (
    LIBCUDF_TO_SUPPORTED_NUMPY_TYPES,
    SUPPORTED_NUMPY_TO_LIBCUDF_TYPES,
    NullHandling,
)
from cudf.utils import cudautils

from cudf._lib.types cimport (
    underlying_type_t_interpolation,
    underlying_type_t_null_policy,
    underlying_type_t_type_id,
)

from numba.np import numpy_support

from cudf._lib.types import Interpolation

cimport cudf._lib.cpp.aggregation as libcudf_aggregation
cimport cudf._lib.cpp.types as libcudf_types

import cudf


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
    VAR = libcudf_aggregation.aggregation.Kind.VARIANCE
    STD = libcudf_aggregation.aggregation.Kind.STD
    MEDIAN = libcudf_aggregation.aggregation.Kind.MEDIAN
    QUANTILE = libcudf_aggregation.aggregation.Kind.QUANTILE
    ARGMAX = libcudf_aggregation.aggregation.Kind.ARGMAX
    ARGMIN = libcudf_aggregation.aggregation.Kind.ARGMIN
    NUNIQUE = libcudf_aggregation.aggregation.Kind.NUNIQUE
    NTH = libcudf_aggregation.aggregation.Kind.NTH_ELEMENT
    COLLECT = libcudf_aggregation.aggregation.Kind.COLLECT
    UNIQUE = libcudf_aggregation.aggregation.Kind.COLLECT_SET
    PTX = libcudf_aggregation.aggregation.Kind.PTX
    CUDA = libcudf_aggregation.aggregation.Kind.CUDA


cdef class Aggregation:
    """A Cython wrapper for aggregations.

    **This class should never be instantiated using a standard constructor,
    only using one of its many factories.** These factories handle mapping
    different cudf operations to their libcudf analogs, e.g.
    `cudf.DataFrame.idxmin` -> `libcudf.argmin`. Additionally, they perform
    any additional configuration needed to translate Python arguments into
    their corresponding C++ types (for instance, C++ enumerations used for
    flag arguments). The factory approach is necessary to support operations
    like `df.agg(lambda x: x.sum())`; such functions are called with this
    class as an argument to generation the desired aggregation.
    """
    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name

    @classmethod
    def sum(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_sum_aggregation[aggregation]())
        return agg

    @classmethod
    def min(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_min_aggregation[aggregation]())
        return agg

    @classmethod
    def max(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_max_aggregation[aggregation]())
        return agg

    @classmethod
    def idxmin(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmin_aggregation[aggregation]())
        return agg

    @classmethod
    def idxmax(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmax_aggregation[aggregation]())
        return agg

    @classmethod
    def mean(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_mean_aggregation[aggregation]())
        return agg

    @classmethod
    def count(cls, dropna=True):
        cdef libcudf_types.null_policy c_null_handling
        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[aggregation](
                c_null_handling
            ))
        return agg

    @classmethod
    def size(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[aggregation](
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.INCLUDE
                )
            ))
        return agg

    @classmethod
    def nunique(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_nunique_aggregation[aggregation]())
        return agg

    @classmethod
    def nth(cls, libcudf_types.size_type size):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_nth_element_aggregation[aggregation](
                size))
        return agg

    @classmethod
    def first(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_nth_element_aggregation[aggregation](
                0,
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return agg

    @classmethod
    def last(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_nth_element_aggregation[aggregation](
                -1,
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return agg

    @classmethod
    def any(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_any_aggregation[aggregation]())
        return agg

    @classmethod
    def all(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_all_aggregation[aggregation]())
        return agg

    @classmethod
    def product(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_product_aggregation[aggregation]())
        return agg
    prod = product

    @classmethod
    def sum_of_squares(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_sum_of_squares_aggregation[aggregation]()
        )
        return agg

    @classmethod
    def var(cls, ddof=1):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_variance_aggregation[aggregation](ddof))
        return agg

    @classmethod
    def std(cls, ddof=1):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_std_aggregation[aggregation](ddof))
        return agg

    @classmethod
    def median(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_median_aggregation[aggregation]())
        return agg

    @classmethod
    def quantile(cls, q=0.5, interpolation="linear"):
        cdef Aggregation agg = cls()

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
            libcudf_aggregation.make_quantile_aggregation[aggregation](
                c_q, c_interp)
        )
        return agg

    @classmethod
    def collect(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_collect_list_aggregation[aggregation]())
        return agg

    @classmethod
    def unique(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_collect_set_aggregation[aggregation]())
        return agg

    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        cdef Aggregation agg = cls()

        cdef libcudf_types.type_id tid
        cdef libcudf_types.data_type out_dtype
        cdef string cpp_str

        # Handling UDF type
        nb_type = numpy_support.from_dtype(kwargs['dtype'])
        type_signature = (nb_type[:],)
        compiled_op = cudautils.compile_udf(op, type_signature)
        output_np_dtype = cudf.dtype(compiled_op[1])
        cpp_str = compiled_op[0].encode('UTF-8')
        if output_np_dtype not in SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
            raise TypeError(
                "Result of window function has unsupported dtype {}"
                .format(op[1])
            )
        tid = (
            <libcudf_types.type_id> (
                <underlying_type_t_type_id> (
                    SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[output_np_dtype]
                )
            )
        )
        out_dtype = libcudf_types.data_type(tid)

        agg.c_obj = move(
            libcudf_aggregation.make_udf_aggregation[aggregation](
                libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
            ))
        return agg

    # scan aggregations
    # TODO: update this after adding per algorithm aggregation derived types
    # https://github.com/rapidsai/cudf/issues/7106
    cumsum = sum
    cummin = min
    cummax = max

    @classmethod
    def cumcount(cls):
        cdef Aggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[aggregation](
                libcudf_types.null_policy.INCLUDE
            ))
        return agg

cdef class RollingAggregation:
    """A Cython wrapper for rolling window aggregations.

    **This class should never be instantiated using a standard constructor,
    only using one of its many factories.** These factories handle mapping
    different cudf operations to their libcudf analogs, e.g.
    `cudf.DataFrame.idxmin` -> `libcudf.argmin`. Additionally, they perform
    any additional configuration needed to translate Python arguments into
    their corresponding C++ types (for instance, C++ enumerations used for
    flag arguments). The factory approach is necessary to support operations
    like `df.agg(lambda x: x.sum())`; such functions are called with this
    class as an argument to generation the desired aggregation.
    """
    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name

    @classmethod
    def sum(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_sum_aggregation[rolling_aggregation]())
        return agg

    @classmethod
    def min(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_min_aggregation[rolling_aggregation]())
        return agg

    @classmethod
    def max(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_max_aggregation[rolling_aggregation]())
        return agg

    @classmethod
    def idxmin(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmin_aggregation[
                rolling_aggregation]())
        return agg

    @classmethod
    def idxmax(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmax_aggregation[
                rolling_aggregation]())
        return agg

    @classmethod
    def mean(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_mean_aggregation[rolling_aggregation]())
        return agg

    @classmethod
    def var(cls, ddof=1):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_variance_aggregation[rolling_aggregation](
                ddof
            )
        )
        return agg

    @classmethod
    def std(cls, ddof=1):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_std_aggregation[rolling_aggregation](ddof)
        )
        return agg

    @classmethod
    def count(cls, dropna=True):
        cdef libcudf_types.null_policy c_null_handling
        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[rolling_aggregation](
                c_null_handling
            ))
        return agg

    @classmethod
    def size(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[rolling_aggregation](
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.INCLUDE)
            ))
        return agg

    @classmethod
    def collect(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_collect_list_aggregation[
                rolling_aggregation]())
        return agg

    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        cdef RollingAggregation agg = cls()

        cdef libcudf_types.type_id tid
        cdef libcudf_types.data_type out_dtype
        cdef string cpp_str

        # Handling UDF type
        nb_type = numpy_support.from_dtype(kwargs['dtype'])
        type_signature = (nb_type[:],)
        compiled_op = cudautils.compile_udf(op, type_signature)
        output_np_dtype = cudf.dtype(compiled_op[1])
        cpp_str = compiled_op[0].encode('UTF-8')
        if output_np_dtype not in SUPPORTED_NUMPY_TO_LIBCUDF_TYPES:
            raise TypeError(
                "Result of window function has unsupported dtype {}"
                .format(op[1])
            )
        tid = (
            <libcudf_types.type_id> (
                <underlying_type_t_type_id> (
                    SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[output_np_dtype]
                )
            )
        )
        out_dtype = libcudf_types.data_type(tid)

        agg.c_obj = move(
            libcudf_aggregation.make_udf_aggregation[rolling_aggregation](
                libcudf_aggregation.udf_type.PTX, cpp_str, out_dtype
            ))
        return agg

    # scan aggregations
    # TODO: update this after adding per algorithm aggregation derived types
    # https://github.com/rapidsai/cudf/issues/7106
    cumsum = sum
    cummin = min
    cummax = max

    @classmethod
    def cumcount(cls):
        cdef RollingAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[rolling_aggregation](
                libcudf_types.null_policy.INCLUDE
            ))
        return agg

cdef class GroupbyAggregation:
    """A Cython wrapper for groupby aggregations.

    **This class should never be instantiated using a standard constructor,
    only using one of its many factories.** These factories handle mapping
    different cudf operations to their libcudf analogs, e.g.
    `cudf.DataFrame.idxmin` -> `libcudf.argmin`. Additionally, they perform
    any additional configuration needed to translate Python arguments into
    their corresponding C++ types (for instance, C++ enumerations used for
    flag arguments). The factory approach is necessary to support operations
    like `df.agg(lambda x: x.sum())`; such functions are called with this
    class as an argument to generation the desired aggregation.
    """
    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name

    @classmethod
    def sum(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_sum_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def min(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_min_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def max(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_max_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def idxmin(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmin_aggregation[
                groupby_aggregation]())
        return agg

    @classmethod
    def idxmax(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_argmax_aggregation[
                groupby_aggregation]())
        return agg

    @classmethod
    def mean(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_mean_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def count(cls, dropna=True):
        cdef libcudf_types.null_policy c_null_handling
        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[groupby_aggregation](
                c_null_handling
            ))
        return agg

    @classmethod
    def size(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.make_count_aggregation[groupby_aggregation](
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.INCLUDE)
            ))
        return agg

    @classmethod
    def collect(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_collect_list_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def nunique(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_nunique_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def nth(cls, libcudf_types.size_type size):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](size))
        return agg

    @classmethod
    def product(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_product_aggregation[groupby_aggregation]())
        return agg
    prod = product

    @classmethod
    def sum_of_squares(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_sum_of_squares_aggregation[groupby_aggregation]()
        )
        return agg

    @classmethod
    def var(cls, ddof=1):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_variance_aggregation[groupby_aggregation](ddof))
        return agg

    @classmethod
    def std(cls, ddof=1):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_std_aggregation[groupby_aggregation](ddof))
        return agg

    @classmethod
    def median(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_median_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def quantile(cls, q=0.5, interpolation="linear"):
        cdef GroupbyAggregation agg = cls()

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
            libcudf_aggregation.make_quantile_aggregation[groupby_aggregation](
                c_q, c_interp)
        )
        return agg

    @classmethod
    def unique(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_collect_set_aggregation[groupby_aggregation]())
        return agg

    @classmethod
    def first(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](
                0,
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return agg

    @classmethod
    def last(cls):
        cdef GroupbyAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](
                -1,
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return agg

cdef class GroupbyScanAggregation:
    """A Cython wrapper for groupby scan aggregations.

    **This class should never be instantiated using a standard constructor,
    only using one of its many factories.** These factories handle mapping
    different cudf operations to their libcudf analogs, e.g.
    `cudf.DataFrame.idxmin` -> `libcudf.argmin`. Additionally, they perform
    any additional configuration needed to translate Python arguments into
    their corresponding C++ types (for instance, C++ enumerations used for
    flag arguments). The factory approach is necessary to support operations
    like `df.agg(lambda x: x.sum())`; such functions are called with this
    class as an argument to generation the desired aggregation.
    """
    @property
    def kind(self):
        return AggregationKind(self.c_obj.get()[0].kind).name

    @classmethod
    def sum(cls):
        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_sum_aggregation[groupby_scan_aggregation]())
        return agg

    @classmethod
    def min(cls):
        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_min_aggregation[groupby_scan_aggregation]())
        return agg

    @classmethod
    def max(cls):
        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_max_aggregation[groupby_scan_aggregation]())
        return agg

    @classmethod
    def count(cls, dropna=True):
        cdef libcudf_types.null_policy c_null_handling
        if dropna:
            c_null_handling = libcudf_types.null_policy.EXCLUDE
        else:
            c_null_handling = libcudf_types.null_policy.INCLUDE

        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_count_aggregation[groupby_scan_aggregation](c_null_handling))
        return agg

    @classmethod
    def size(cls):
        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_count_aggregation[groupby_scan_aggregation](
                <libcudf_types.null_policy><underlying_type_t_null_policy>(
                    NullHandling.INCLUDE)
            ))
        return agg

    @classmethod
    def cumcount(cls):
        cdef GroupbyScanAggregation agg = cls()
        agg.c_obj = move(
            libcudf_aggregation.
            make_count_aggregation[groupby_scan_aggregation](
                libcudf_types.null_policy.INCLUDE
            ))
        return agg

    # scan aggregations
    # TODO: update this after adding per algorithm aggregation derived types
    # https://github.com/rapidsai/cudf/issues/7106
    cumsum = sum
    cummin = min
    cummax = max


cdef Aggregation make_aggregation(op, kwargs=None):
    r"""
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
    \*\*kwargs : dict, optional
        Any keyword arguments to be passed to the op.

    Returns
    -------
    Aggregation
    """
    if kwargs is None:
        kwargs = {}

    cdef Aggregation agg
    if isinstance(op, str):
        agg = getattr(Aggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            agg = Aggregation.collect()
        elif "dtype" in kwargs:
            agg = Aggregation.from_udf(op, **kwargs)
        else:
            agg = op(Aggregation)
    else:
        raise TypeError(f"Unknown aggregation {op}")
    return agg

cdef RollingAggregation make_rolling_aggregation(op, kwargs=None):
    r"""
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
    \*\*kwargs : dict, optional
        Any keyword arguments to be passed to the op.

    Returns
    -------
    RollingAggregation
    """
    if kwargs is None:
        kwargs = {}

    cdef RollingAggregation agg
    if isinstance(op, str):
        agg = getattr(RollingAggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            agg = RollingAggregation.collect()
        elif "dtype" in kwargs:
            agg = RollingAggregation.from_udf(op, **kwargs)
        else:
            agg = op(RollingAggregation)
    else:
        raise TypeError(f"Unknown aggregation {op}")
    return agg

cdef GroupbyAggregation make_groupby_aggregation(op, kwargs=None):
    r"""
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
    \*\*kwargs : dict, optional
        Any keyword arguments to be passed to the op.

    Returns
    -------
    GroupbyAggregation
    """
    if kwargs is None:
        kwargs = {}

    cdef GroupbyAggregation agg
    if isinstance(op, str):
        agg = getattr(GroupbyAggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            agg = GroupbyAggregation.collect()
        elif "dtype" in kwargs:
            agg = GroupbyAggregation.from_udf(op, **kwargs)
        else:
            agg = op(GroupbyAggregation)
    else:
        raise TypeError(f"Unknown aggregation {op}")
    return agg

cdef GroupbyScanAggregation make_groupby_scan_aggregation(op, kwargs=None):
    r"""
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
    \*\*kwargs : dict, optional
        Any keyword arguments to be passed to the op.

    Returns
    -------
    GroupbyScanAggregation
    """
    if kwargs is None:
        kwargs = {}

    cdef GroupbyScanAggregation agg
    if isinstance(op, str):
        agg = getattr(GroupbyScanAggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            agg = GroupbyScanAggregation.collect()
        elif "dtype" in kwargs:
            agg = GroupbyScanAggregation.from_udf(op, **kwargs)
        else:
            agg = op(GroupbyScanAggregation)
    else:
        raise TypeError(f"Unknown aggregation {op}")
    return agg
