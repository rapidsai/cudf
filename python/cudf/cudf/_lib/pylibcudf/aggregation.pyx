# Copyright (c) 2023, NVIDIA CORPORATION.

from enum import IntEnum

from libcpp.vector cimport vector

from cudf._lib.cpp cimport aggregation as libcudf_aggregation
from cudf._lib.cpp.aggregation cimport (
    groupby_aggregation,
    make_all_aggregation,
    make_any_aggregation,
    make_argmax_aggregation,
    make_argmin_aggregation,
    make_collect_list_aggregation,
    make_collect_set_aggregation,
    make_count_aggregation,
    make_covariance_aggregation,
    make_max_aggregation,
    make_mean_aggregation,
    make_median_aggregation,
    make_min_aggregation,
    make_nth_element_aggregation,
    make_nunique_aggregation,
    make_product_aggregation,
    make_quantile_aggregation,
    make_rank_aggregation,
    make_std_aggregation,
    make_sum_aggregation,
    make_udf_aggregation,
    make_variance_aggregation,
)
from cudf._lib.cpp.types cimport interpolation as interpolation_t, size_type

from cudf._lib.types import Interpolation, NullHandling

from cudf._lib.types cimport underlying_type_t_interpolation

from .types cimport null_policy, underlying_type_t_null_policy

import pandas as pd

from cudf._lib.cpp.aggregation cimport (
    underlying_type_t_correlation_type,
    underlying_type_t_rank_method,
)


# In libcudf we benefit from compile-time checks around which types of
# aggregations are valid via the multiple inheritance patterns used by
# aggregation types. In Python code we cannot get the same benefits, however,
# so we also don't need to mirror the same inheritance structures and classes.
cdef class Aggregation:
    def __init__(self):
        raise ValueError(
            "Aggregation types should not be constructed directly. "
            "Use one of the factory functions."
        )

class CorrelationType(IntEnum):
    PEARSON = (
        <underlying_type_t_correlation_type>
        libcudf_aggregation.correlation_type.PEARSON
    )
    KENDALL = (
        <underlying_type_t_correlation_type>
        libcudf_aggregation.correlation_type.KENDALL
    )
    SPEARMAN = (
        <underlying_type_t_correlation_type>
        libcudf_aggregation.correlation_type.SPEARMAN
    )



cdef class GroupbyAggregation(Aggregation):
    # TODO: We need to decide what we actually want this API to look like. The
    # inability to have cpdef classmethods here is pretty restrictive because
    # it really is the most natural API for factories e.g.
    # GroupbyAggregation.sum(...).  For now cpdef free functions may be the
    # best that we can do but it really isn't as readable to have something
    # like groupbyaggregation_sum(...) (or any other variation `make_*` etc)
    @classmethod
    def sum(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_sum_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def min(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_min_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def max(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_max_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def idxmin(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_argmin_aggregation[
                groupby_aggregation]())
        return obj

    @classmethod
    def idxmax(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_argmax_aggregation[
                groupby_aggregation]())
        return obj

    @classmethod
    def mean(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_mean_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def count(cls, dropna=True):
        cdef null_policy c_null_handling
        if dropna:
            c_null_handling = null_policy.EXCLUDE
        else:
            c_null_handling = null_policy.INCLUDE

        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_count_aggregation[groupby_aggregation](
                c_null_handling
            ))
        return obj

    @classmethod
    def size(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.make_count_aggregation[groupby_aggregation](
                <null_policy><underlying_type_t_null_policy>(
                    NullHandling.INCLUDE)
            ))
        return obj

    @classmethod
    def collect(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_collect_list_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def nunique(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_nunique_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def nth(cls, size_type size):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](size))
        return obj

    @classmethod
    def product(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_product_aggregation[groupby_aggregation]())
        return obj
    prod = product

    @classmethod
    def sum_of_squares(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_sum_of_squares_aggregation[groupby_aggregation]()
        )
        return obj

    @classmethod
    def var(cls, ddof=1):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_variance_aggregation[groupby_aggregation](ddof))
        return obj

    @classmethod
    def std(cls, ddof=1):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_std_aggregation[groupby_aggregation](ddof))
        return obj

    @classmethod
    def median(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_median_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def quantile(cls, q=0.5, interpolation="linear"):
        cdef GroupbyAggregation obj = cls.__new__()

        if not pd.api.types.is_list_like(q):
            q = [q]

        cdef vector[double] c_q = q
        cdef interpolation_t c_interp = (
            <interpolation_t> (
                <underlying_type_t_interpolation> (
                    Interpolation[interpolation.upper()]
                )
            )
        )
        obj.c_obj.swap(
            libcudf_aggregation.make_quantile_aggregation[groupby_aggregation](
                c_q, c_interp)
        )
        return obj

    @classmethod
    def unique(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_collect_set_aggregation[groupby_aggregation]())
        return obj

    @classmethod
    def first(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](
                0,
                <null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return obj

    @classmethod
    def last(cls):
        cdef GroupbyAggregation obj = cls.__new__()
        obj.c_obj.swap(
            libcudf_aggregation.
            make_nth_element_aggregation[groupby_aggregation](
                -1,
                <null_policy><underlying_type_t_null_policy>(
                    NullHandling.EXCLUDE
                )
            )
        )
        return obj

    @classmethod
    def corr(cls, method, size_type min_periods):
        cdef GroupbyAggregation obj = cls.__new__()
        cdef libcudf_aggregation.correlation_type c_method = (
            <libcudf_aggregation.correlation_type> (
                <underlying_type_t_correlation_type> (
                    CorrelationType[method.upper()]
                )
            )
        )
        obj.c_obj.swap(
            libcudf_aggregation.
            make_correlation_aggregation[groupby_aggregation](
                c_method, min_periods
            ))
        return obj

    @classmethod
    def cov(
        cls,
        size_type min_periods,
        size_type ddof=1
    ):
        cdef GroupbyAggregation obj = cls.__new__()

        obj.c_obj.swap(
            libcudf_aggregation.
            make_covariance_aggregation[groupby_aggregation](
                min_periods, ddof
            ))
        return obj

    cdef groupby_aggregation * get(self) nogil:
        """Get the underlying agg object."""
        return self.c_obj.get()
