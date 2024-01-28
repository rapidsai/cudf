# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# Have to alias this so that the Pythonic name gets capitalized
# TODO: Expose the enums to Python
from cudf._lib.cpp.aggregation cimport (
    Kind as kind_t,
    aggregation,
    correlation_type,
    groupby_aggregation,
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

# Notes:
# - We use raw pointers because Cython does not understand converting between
#   compatible unique_ptr types
# - We invert the inheritance hierarchy from the C++ side. In C++, the
#   operation classes inherit from the algorithm classes, which are effectively
#   used as tags to determine which operations are supported. This pattern is
#   useful in C++ because it provides compile-time checks on what can be
#   constructed. In Python we cannot leverage this information though since we
#   work with a precompiled libcudf library so the symbols are fully
#   predetermined and the template definitions are not available. Moreover,
#   Cython's knowledge of templating is insufficient. Therefore, we have to
#   manage the set of supported operations by algorithm. The inverted hierarchy
#   and use of mixins is the cleanest approach for doing that.


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

    cdef unique_ptr[groupby_aggregation] make_groupby_copy(self) except *:
        """Make a copy of the aggregation that can be used in a groupby.

        This function will raise an exception if the aggregation is not supported
        as a groupby aggregation.
        """
        cdef unique_ptr[aggregation] agg = self.c_ptr.clone()
        # This roundabout casting is required because Cython does not understand
        # that unique_ptrs can be cast along class hierarchies.
        cdef aggregation *raw_agg = agg.get()
        cdef groupby_aggregation *agg_cast = dynamic_cast[gba_ptr](raw_agg)
        if agg_cast is NULL:
            agg_repr = str(self.kind()).split(".")[1].title()
            raise TypeError(f"{agg_repr} aggregations are not supported by groupby")
        agg.release()
        return unique_ptr[groupby_aggregation](agg_cast)

    # TODO: If/when https://github.com/cython/cython/issues/1271 is resolved, we
    # should migrate the various factories to cpdef classmethods instead of Python
    # def methods.
    @classmethod
    def sum(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_sum_aggregation[aggregation]
        )

    @classmethod
    def product(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_product_aggregation[aggregation]
        )

    @classmethod
    def min(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_min_aggregation[aggregation]
        )

    @classmethod
    def max(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_max_aggregation[aggregation]
        )

    @classmethod
    def count(cls, null_policy null_handling):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_count_aggregation[aggregation],
            null_handling,
        )

    @classmethod
    def any(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_any_aggregation[aggregation],
        )

    @classmethod
    def all(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_all_aggregation[aggregation]
        )

    @classmethod
    def sum_of_squares(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_sum_of_squares_aggregation[aggregation],
        )

    @classmethod
    def mean(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_mean_aggregation[aggregation],
        )

    @classmethod
    def variance(cls, size_type ddof):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_variance_aggregation[aggregation](ddof))
        out.c_ptr = ptr.release()
        return out

    @classmethod
    def std(cls, size_type ddof):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_std_aggregation[aggregation](ddof))
        out.c_ptr = ptr.release()
        return out

    @classmethod
    def median(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_median_aggregation[aggregation]
        )

    @classmethod
    def quantile(cls, list quantiles, interpolation interp = interpolation.LINEAR):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_quantile_aggregation[aggregation](quantiles, interp))
        out.c_ptr = ptr.release()
        return out

    @classmethod
    def argmax(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_argmax_aggregation[aggregation]
        )

    @classmethod
    def argmin(cls):
        return _create_nullary_agg(
            cls, cpp_aggregation.make_argmin_aggregation[aggregation]
        )

    @classmethod
    def nunique(cls, null_policy null_handling = null_policy.EXCLUDE):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_nunique_aggregation[aggregation],
            null_handling,
        )

    @classmethod
    def nth_element(cls, size_type n, null_policy null_handling = null_policy.INCLUDE):
        cdef Aggregation out = cls.__new__(cls)
        cdef unique_ptr[aggregation] ptr = move(
            cpp_aggregation.make_nth_element_aggregation[aggregation](n, null_handling))
        out.c_ptr = ptr.release()
        return out

    @classmethod
    def collect_list(cls, null_policy null_handling = null_policy.INCLUDE):
        return _create_unary_null_handling_agg(
            cls,
            cpp_aggregation.make_collect_list_aggregation[aggregation],
            null_handling,
        )

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

#     @classmethod
#     def udf(
#         cls,
#         udf_type type,
#         string user_defined_aggregator,
#         data_type output_type
#     ):

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
