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

    cpdef kind_t kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_ptr).kind

    cdef unique_ptr[groupby_aggregation] clone_as_groupby(self) except *:
        """Make a copy of the aggregation that can be used in a groupby.

        This function will raise an exception if the aggregation is not supported
        as a groupby aggregation.
        """
        cdef unique_ptr[aggregation] agg = dereference(self.c_ptr).clone()
        # This roundabout casting is required because Cython does not understand
        # that unique_ptrs can be cast along class hierarchies.
        cdef aggregation *raw_agg = agg.get()
        cdef groupby_aggregation *agg_cast = dynamic_cast[gba_ptr](raw_agg)
        if agg_cast is NULL:
            agg_repr = str(self.kind()).split(".")[1].title()
            raise TypeError(f"{agg_repr} aggregations are not supported by groupby")
        agg.release()
        return unique_ptr[groupby_aggregation](agg_cast)

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg):
        """Create a Python Aggregation from a libcudf aggregation."""
        cdef Aggregation out = Aggregation.__new__(Aggregation)
        out.c_ptr = move(agg)
        return out


cpdef Aggregation sum():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_sum_aggregation[aggregation]())
    )


cpdef Aggregation product():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_product_aggregation[aggregation]())
    )


cpdef Aggregation min():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_min_aggregation[aggregation]())
    )


cpdef Aggregation max():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_max_aggregation[aggregation]())
    )


cpdef Aggregation count(null_policy null_handling):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_count_aggregation[aggregation](null_handling))
    )


cpdef Aggregation any():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_any_aggregation[aggregation]())
    )


cpdef Aggregation all():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_all_aggregation[aggregation]())
    )


cpdef Aggregation sum_of_squares():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_sum_of_squares_aggregation[aggregation]())
    )


cpdef Aggregation mean():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_mean_aggregation[aggregation]())
    )


cpdef Aggregation variance(size_type ddof):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_variance_aggregation[aggregation](ddof))
    )


cpdef Aggregation std(size_type ddof):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_std_aggregation[aggregation](ddof))
    )


cpdef Aggregation median():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_median_aggregation[aggregation]())
    )


cpdef Aggregation quantile(list quantiles, interpolation interp = interpolation.LINEAR):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_quantile_aggregation[aggregation](quantiles, interp))
    )


cpdef Aggregation argmax():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_argmax_aggregation[aggregation]())
    )


cpdef Aggregation argmin():
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_argmin_aggregation[aggregation]())
    )


cpdef Aggregation nunique(null_policy null_handling = null_policy.EXCLUDE):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_nunique_aggregation[aggregation](null_handling))
    )


cpdef Aggregation nth_element(
    size_type n, null_policy null_handling = null_policy.INCLUDE
):
    return Aggregation.from_libcudf(
        move(
            cpp_aggregation.make_nth_element_aggregation[aggregation](n, null_handling)
        )
    )


cpdef Aggregation collect_list(null_policy null_handling = null_policy.INCLUDE):
    return Aggregation.from_libcudf(
        move(cpp_aggregation.make_collect_list_aggregation[aggregation](null_handling))
    )


cpdef Aggregation collect_set(
    null_handling = null_policy.INCLUDE,
    nulls_equal = null_equality.EQUAL,
    nans_equal = nan_equality.ALL_EQUAL,
):
    return Aggregation.from_libcudf(
        move(
            cpp_aggregation.make_collect_set_aggregation[aggregation](
                null_handling, nulls_equal, nans_equal
            )
        )
    )

# cpdef Aggregation udf(
#     udf_type type,
#     string user_defined_aggregator,
#     data_type output_type
# ):


cpdef Aggregation correlation(correlation_type type, size_type min_periods):
    return Aggregation.from_libcudf(
        move(
            cpp_aggregation.make_correlation_aggregation[aggregation](
                type, min_periods
            )
        )
    )


cpdef Aggregation covariance(size_type min_periods, size_type ddof):
    return Aggregation.from_libcudf(
        move(
            cpp_aggregation.make_covariance_aggregation[aggregation](
                min_periods, ddof
            )
        )
    )


cpdef Aggregation rank(
    rank_method method,
    order column_order = order.ASCENDING,
    null_policy null_handling = null_policy.EXCLUDE,
    null_order null_precedence = null_order.AFTER,
    rank_percentage percentage = rank_percentage.NONE,
):
    return Aggregation.from_libcudf(
        move(
            cpp_aggregation.make_rank_aggregation[aggregation](
                method,
                column_order,
                null_handling,
                null_precedence,
                percentage,
            )
        )
    )
