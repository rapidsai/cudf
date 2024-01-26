# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# Have to alias this so that the Pythonic name gets capitalized
# TODO: Expose the enums to Python
from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    correlation_type,
    rank_method,
    rank_percentage,
)
from cudf._lib.cpp.types cimport (
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)


cdef class Aggregation:
    """A wrapper for aggregations.

    **Neither this class nor any of its subclasses should ever be instantiated
    using a standard constructor, only using one of its many factories.** The
    factory approach matches the libcudf approach, which is necessary because
    some aggregations require additional arguments beyond the kind.
    """
    def __init__(self):
        raise ValueError(
            "Aggregation types should not be constructed directly. "
            "Use one of the factory functions."
        )

    def __dealloc__(self):
        # Deletion is always handled by child classes.
        if type(self) is not Aggregation and self.c_ptr is not NULL:
            del self.c_ptr

    cpdef kind_t kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_ptr).kind


cdef Aggregation _create_nullary_agg(cls, nullary_factory_type cpp_agg_factory):
    cdef Aggregation out = cls.__new__(cls)
    cdef unique_ptr[aggregation] ptr = move(cpp_agg_factory())
    out.c_ptr = ptr.release()
    return out


cdef Aggregation _create_unary_null_handling_agg(
    cls,
    unary_null_handling_factory_type cpp_agg_factory,
    null_policy null_handling,
):
    cdef Aggregation out = cls.__new__(cls)
    cdef unique_ptr[aggregation] ptr = move(cpp_agg_factory(null_handling))
    out.c_ptr = ptr.release()
    return out


# TODO: If/when https://github.com/cython/cython/issues/1271 is resolved, we
# should migrate the various factories to cpdef classmethods instead of Python
# def methods.

# The below aggregation types are effectively mixins used to implement specific
# types of aggregations. They function like mixins when combined with the
# algorithm-specific classes below. The factory methods all construct an object
# of type cls but save that to a variable of the base class Aggregation,
# allowing the logic of the function to be generic regardless of which concrete
# algorithm-type the mixin is mixed into. Since the methods are pure Python,
# the end result is an object that Python will dynamically resolve to the
# correct type at the call site.
class SumAggregation(Aggregation):
    @classmethod
    def sum(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_sum_aggregation[aggregation]
        )


class ProductAggregation(Aggregation):
    @classmethod
    def product(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_product_aggregation[aggregation]
        )


class MinAggregation(Aggregation):
    @classmethod
    def min(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_min_aggregation[aggregation]
        )


class MaxAggregation(Aggregation):
    @classmethod
    def max(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_max_aggregation[aggregation]
        )


class CountAggregation(Aggregation):
    @classmethod
    def count(cls, null_policy null_handling):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_count_aggregation[aggregation],
            null_handling,
        )


class AnyAggregation(Aggregation):
    @classmethod
    def any(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_any_aggregation[aggregation],
        )


class AllAggregation(Aggregation):
    @classmethod
    def all(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_all_aggregation[aggregation]
        )


class SumOfSquaresAggregation(Aggregation):
    @classmethod
    def sum_of_squares(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_sum_of_squares_aggregation[aggregation],
        )


class MeanAggregation(Aggregation):
    @classmethod
    def mean(cls):
        return _create_nullary_agg(
            cls,
            cpp_aggregation.make_mean_aggregation[aggregation],
        )


class VarianceAggregation(Aggregation):
    @classmethod
    def variance(cls, size_type ddof):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_variance_aggregation[aggregation](ddof))
        out.c_ptr = ptr.release()
        return out


class StdAggregation(Aggregation):
    @classmethod
    def std(cls, size_type ddof):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_std_aggregation[aggregation](ddof))
        out.c_ptr = ptr.release()
        return out


class MedianAggregation(Aggregation):
    @classmethod
    def median(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_median_aggregation[aggregation]
        )


class QuantileAggregation(Aggregation):
    @classmethod
    def quantile(cls, list quantiles, interpolation interp = interpolation.LINEAR):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_quantile_aggregation[aggregation](quantiles, interp))
        out.c_ptr = ptr.release()
        return out


class ArgmaxAggregation(Aggregation):
    @classmethod
    def argmax(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_argmax_aggregation[aggregation]
        )


class ArgminAggregation(Aggregation):
    @classmethod
    def argmin(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_argmin_aggregation[aggregation]
        )


class NuniqueAggregation(Aggregation):
    @classmethod
    def nunique(cls, null_policy null_handling = null_policy.EXCLUDE):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_nunique_aggregation[aggregation],
            null_handling,
        )


class NthElementAggregation(Aggregation):
    @classmethod
    def nth_element(cls, size_type n, null_policy null_handling = null_policy.INCLUDE):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_nth_element_aggregation[aggregation](n, null_handling))
        out.c_ptr = ptr.release()
        return out


class CollectListAggregation(Aggregation):
    @classmethod
    def collect_list(cls, null_policy null_handling = null_policy.INCLUDE):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_collect_list_aggregation[aggregation],
            null_handling,
        )


class CollectSetAggregation(Aggregation):
    @classmethod
    def collect_set(
        cls,
        null_handling = null_policy.INCLUDE,
        nulls_equal = null_equality.EQUAL,
        nans_equal = nan_equality.ALL_EQUAL,
    ):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_collect_set_aggregation[aggregation](
                null_handling, nulls_equal, nans_equal
            ))
        out.c_ptr = ptr.release()
        return out


# class UdfAggregation(Aggregation):
#     @classmethod
#     def udf(
#         cls,
#         udf_type type,
#         string user_defined_aggregator,
#         data_type output_type
#     ):


class CorrelationAggregation(Aggregation):
    @classmethod
    def correlation(cls, correlation_type type, size_type min_periods):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_correlation_aggregation[aggregation](
                type, min_periods
            )
        )
        out.c_ptr = ptr.release()
        return out


class CovarianceAggregation(Aggregation):
    @classmethod
    def covariance(cls, size_type min_periods, size_type ddof):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_covariance_aggregation[aggregation](
                min_periods, ddof
            )
        )
        out.c_ptr = ptr.release()
        return out


class RankAggregation(Aggregation):
    @classmethod
    def rank(
        cls,
        rank_method method,
        order column_order = order.ASCENDING,
        null_policy null_handling = null_policy.EXCLUDE,
        null_order null_precedence = null_order.AFTER,
        rank_percentage percentage = rank_percentage.NONE,
    ):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_rank_aggregation[aggregation](
                method, column_order, null_handling, null_precedence, percentage
            )
        )
        out.c_ptr = ptr.release()
        return out


# The following are the concrete aggregation types corresponding to aggregation
# algorithms.
class RollingAggregation(
    SumAggregation,
    MinAggregation,
    MaxAggregation,
    CountAggregation,
    MeanAggregation,
    StdAggregation,
    VarianceAggregation,
    ArgmaxAggregation,
    ArgminAggregation,
    NthElementAggregation,
    RankAggregation,
    CollectListAggregation,
    CollectSetAggregation,
):
    pass


class GroupbyAggregation(
    SumAggregation,
    ProductAggregation,
    MinAggregation,
    MaxAggregation,
    CountAggregation,
    SumOfSquaresAggregation,
    MeanAggregation,
    StdAggregation,
    VarianceAggregation,
    MedianAggregation,
    QuantileAggregation,
    ArgmaxAggregation,
    ArgminAggregation,
    NuniqueAggregation,
    NthElementAggregation,
    CollectListAggregation,
    CollectSetAggregation,
    CovarianceAggregation,
    CorrelationAggregation,
):
    pass


class GroupbyScanAggregation(
    SumAggregation,
    MinAggregation,
    MaxAggregation,
    CountAggregation,
    RankAggregation,
):
    pass


class ReduceAggregation(
    SumAggregation,
    ProductAggregation,
    MinAggregation,
    MaxAggregation,
    AnyAggregation,
    AllAggregation,
    SumOfSquaresAggregation,
    MeanAggregation,
    StdAggregation,
    VarianceAggregation,
    MedianAggregation,
    QuantileAggregation,
    NuniqueAggregation,
    NthElementAggregation,
    CollectListAggregation,
    CollectSetAggregation,
):
    pass


class ScanAggregation(
    SumAggregation,
    ProductAggregation,
    MinAggregation,
    MaxAggregation,
    RankAggregation,
):
    pass
