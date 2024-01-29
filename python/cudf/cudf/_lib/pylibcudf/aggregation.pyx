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
    make_all_aggregation,
    make_any_aggregation,
    make_argmax_aggregation,
    make_argmin_aggregation,
    make_collect_list_aggregation,
    make_collect_set_aggregation,
    make_correlation_aggregation,
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
    make_sum_of_squares_aggregation,
    make_variance_aggregation,
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

# workaround for https://github.com/cython/cython/issues/3885
ctypedef groupby_aggregation * gba_ptr


cdef class Aggregation:
    """A wrapper for aggregations.

    **Neither this class nor any of its subclasses should ever be instantiated
    using a standard constructor, only using one of its many factories.** The
    factory approach matches the libcudf approach, which is necessary because
    some aggregations require additional arguments beyond the kind.
    """
    def __init__(self):
        raise ValueError(
            "Aggregations should not be constructed directly. Use one of the factories."
        )

    cpdef kind_t kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_obj).kind

    cdef unique_ptr[groupby_aggregation] clone_underlying_as_groupby(self) except *:
        """Make a copy of the underlying aggregation that can be used in a groupby.

        This function will raise an exception if the aggregation is not supported as a
        groupby aggregation. This failure to cast translates the per-algorithm
        aggregation logic encoded in libcudf's type hierarchy into Python.
        """
        cdef unique_ptr[aggregation] agg = dereference(self.c_obj).clone()
        cdef groupby_aggregation *agg_cast = dynamic_cast[gba_ptr](agg.get())
        if agg_cast is NULL:
            agg_repr = str(self.kind()).split(".")[1].title()
            raise TypeError(f"{agg_repr} aggregations are not supported by groupby")
        agg.release()
        return unique_ptr[groupby_aggregation](agg_cast)

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg):
        """Create a Python Aggregation from a libcudf aggregation."""
        cdef Aggregation out = Aggregation.__new__(Aggregation)
        out.c_obj = move(agg)
        return out


cpdef Aggregation sum():
    return Aggregation.from_libcudf(move(make_sum_aggregation[aggregation]()))


cpdef Aggregation product():
    return Aggregation.from_libcudf(move(make_product_aggregation[aggregation]()))


cpdef Aggregation min():
    return Aggregation.from_libcudf(move(make_min_aggregation[aggregation]()))


cpdef Aggregation max():
    return Aggregation.from_libcudf(move(make_max_aggregation[aggregation]()))


cpdef Aggregation count(null_policy null_handling):
    return Aggregation.from_libcudf(
        move(make_count_aggregation[aggregation](null_handling))
    )


cpdef Aggregation any():
    return Aggregation.from_libcudf(move(make_any_aggregation[aggregation]()))


cpdef Aggregation all():
    return Aggregation.from_libcudf(move(make_all_aggregation[aggregation]()))


cpdef Aggregation sum_of_squares():
    return Aggregation.from_libcudf(
        move(make_sum_of_squares_aggregation[aggregation]())
    )


cpdef Aggregation mean():
    return Aggregation.from_libcudf(move(make_mean_aggregation[aggregation]()))


cpdef Aggregation variance(size_type ddof):
    return Aggregation.from_libcudf(move(make_variance_aggregation[aggregation](ddof)))


cpdef Aggregation std(size_type ddof):
    return Aggregation.from_libcudf(move(make_std_aggregation[aggregation](ddof)))


cpdef Aggregation median():
    return Aggregation.from_libcudf(move(make_median_aggregation[aggregation]()))


cpdef Aggregation quantile(list quantiles, interpolation interp = interpolation.LINEAR):
    return Aggregation.from_libcudf(
        move(make_quantile_aggregation[aggregation](quantiles, interp))
    )


cpdef Aggregation argmax():
    return Aggregation.from_libcudf(move(make_argmax_aggregation[aggregation]()))


cpdef Aggregation argmin():
    return Aggregation.from_libcudf(move(make_argmin_aggregation[aggregation]()))


cpdef Aggregation nunique(null_policy null_handling = null_policy.EXCLUDE):
    return Aggregation.from_libcudf(
        move(make_nunique_aggregation[aggregation](null_handling))
    )


cpdef Aggregation nth_element(
    size_type n, null_policy null_handling = null_policy.INCLUDE
):
    return Aggregation.from_libcudf(
        move(make_nth_element_aggregation[aggregation](n, null_handling))
    )


cpdef Aggregation collect_list(null_policy null_handling = null_policy.INCLUDE):
    return Aggregation.from_libcudf(
        move(make_collect_list_aggregation[aggregation](null_handling))
    )


cpdef Aggregation collect_set(
    null_handling = null_policy.INCLUDE,
    nulls_equal = null_equality.EQUAL,
    nans_equal = nan_equality.ALL_EQUAL,
):
    return Aggregation.from_libcudf(
        move(
            make_collect_set_aggregation[aggregation](
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
        move(make_correlation_aggregation[aggregation](type, min_periods))
    )


cpdef Aggregation covariance(size_type min_periods, size_type ddof):
    return Aggregation.from_libcudf(
        move(make_covariance_aggregation[aggregation](min_periods, ddof))
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
            make_rank_aggregation[aggregation](
                method,
                column_order,
                null_handling,
                null_precedence,
                percentage,
            )
        )
    )
