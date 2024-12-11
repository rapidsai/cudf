# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.cast cimport dynamic_cast
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move
from pylibcudf.libcudf.aggregation cimport (
    aggregation,
    correlation_type,
    ewm_history,
    groupby_aggregation,
    groupby_scan_aggregation,
    make_all_aggregation,
    make_any_aggregation,
    make_argmax_aggregation,
    make_argmin_aggregation,
    make_collect_list_aggregation,
    make_collect_set_aggregation,
    make_correlation_aggregation,
    make_count_aggregation,
    make_covariance_aggregation,
    make_ewma_aggregation,
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
    make_udf_aggregation,
    make_variance_aggregation,
    rank_method,
    rank_percentage,
    reduce_aggregation,
    rolling_aggregation,
    scan_aggregation,
)
from pylibcudf.libcudf.types cimport (
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
    size_type,
)

from pylibcudf.libcudf.aggregation import Kind  # no-cython-lint
from pylibcudf.libcudf.aggregation import \
    correlation_type as CorrelationType  # no-cython-lint
from pylibcudf.libcudf.aggregation import \
    ewm_history as EWMHistory  # no-cython-lint
from pylibcudf.libcudf.aggregation import \
    rank_method as RankMethod  # no-cython-lint
from pylibcudf.libcudf.aggregation import \
    rank_percentage as RankPercentage  # no-cython-lint
from pylibcudf.libcudf.aggregation import udf_type as UdfType  # no-cython-lint

from .types cimport DataType


__all__ = [
    "Aggregation",
    "CorrelationType",
    "EWMHistory",
    "Kind",
    "RankMethod",
    "RankPercentage",
    "UdfType",
    "all",
    "any",
    "argmax",
    "argmin",
    "collect_list",
    "collect_set",
    "correlation",
    "count",
    "covariance",
    "ewma",
    "max",
    "mean",
    "median",
    "min",
    "nth_element",
    "nunique",
    "product",
    "quantile",
    "rank",
    "std",
    "sum",
    "sum_of_squares",
    "udf",
    "variance",
]

cdef class Aggregation:
    """A type of aggregation to perform.

    Aggregations are passed to APIs like
    :py:func:`~pylibcudf.groupby.GroupBy.aggregate` to indicate what
    operations to perform. Using a class for aggregations provides a unified
    API for handling parametrizable aggregations. This class should never be
    instantiated directly, only via one of the factory functions.

    For details, see :cpp:class:`cudf::aggregation`.
    """
    def __init__(self):
        raise ValueError(
            "Aggregations should not be constructed directly. Use one of the factories."
        )

    def __eq__(self, other):
        return type(self) is type(other) and (
            dereference(self.c_obj).is_equal(dereference((<Aggregation>other).c_obj))
        )

    def __hash__(self):
        return dereference(self.c_obj).do_hash()

    # TODO: Ideally we would include the return type here, but we need to do so
    # in a way that Sphinx understands (currently have issues due to
    # https://github.com/cython/cython/issues/5609).
    cpdef kind(self):
        """Get the kind of the aggregation."""
        return dereference(self.c_obj).kind

    cdef void _unsupported_agg_error(self, str alg):
        # Te functions calling this all use a dynamic cast between aggregation types,
        # and the cast returning a null pointer is how we capture whether or not
        # libcudf supports a given aggregation for a particular algorithm.
        agg_repr = str(self.kind()).split(".")[1].title()
        raise TypeError(f"{agg_repr} aggregations are not supported by {alg}")

    cdef unique_ptr[groupby_aggregation] clone_underlying_as_groupby(self) except *:
        """Make a copy of the aggregation that can be used in a groupby."""
        cdef unique_ptr[aggregation] agg = dereference(self.c_obj).clone()
        cdef groupby_aggregation *agg_cast = dynamic_cast[gba_ptr](agg.get())
        if agg_cast is NULL:
            self._unsupported_agg_error("groupby")
        agg.release()
        return unique_ptr[groupby_aggregation](agg_cast)

    cdef unique_ptr[groupby_scan_aggregation] clone_underlying_as_groupby_scan(
        self
    ) except *:
        """Make a copy of the aggregation that can be used in a groupby scan."""
        cdef unique_ptr[aggregation] agg = dereference(self.c_obj).clone()
        cdef groupby_scan_aggregation *agg_cast = dynamic_cast[gbsa_ptr](agg.get())
        if agg_cast is NULL:
            self._unsupported_agg_error("groupby_scan")
        agg.release()
        return unique_ptr[groupby_scan_aggregation](agg_cast)

    cdef const reduce_aggregation* view_underlying_as_reduce(self) except *:
        """View the underlying aggregation as a reduce_aggregation."""
        cdef reduce_aggregation *agg_cast = dynamic_cast[ra_ptr](self.c_obj.get())
        if agg_cast is NULL:
            self._unsupported_agg_error("reduce")
        return agg_cast

    cdef const scan_aggregation* view_underlying_as_scan(self) except *:
        """View the underlying aggregation as a scan_aggregation."""
        cdef scan_aggregation *agg_cast = dynamic_cast[sa_ptr](self.c_obj.get())
        if agg_cast is NULL:
            self._unsupported_agg_error("scan")
        return agg_cast

    cdef const rolling_aggregation* view_underlying_as_rolling(self) except *:
        """View the underlying aggregation as a rolling_aggregation."""
        cdef rolling_aggregation *agg_cast = dynamic_cast[roa_ptr](self.c_obj.get())
        if agg_cast is NULL:
            self._unsupported_agg_error("rolling")
        return agg_cast

    @staticmethod
    cdef Aggregation from_libcudf(unique_ptr[aggregation] agg):
        """Create a Python Aggregation from a libcudf aggregation."""
        cdef Aggregation out = Aggregation.__new__(Aggregation)
        out.c_obj = move(agg)
        return out


cpdef Aggregation sum():
    """Create a sum aggregation.

    For details, see :cpp:func:`make_sum_aggregation`.

    Returns
    -------
    Aggregation
        The sum aggregation.
    """
    return Aggregation.from_libcudf(move(make_sum_aggregation[aggregation]()))


cpdef Aggregation product():
    """Create a product aggregation.

    For details, see :cpp:func:`make_product_aggregation`.

    Returns
    -------
    Aggregation
        The product aggregation.
    """
    return Aggregation.from_libcudf(move(make_product_aggregation[aggregation]()))


cpdef Aggregation min():
    """Create a min aggregation.

    For details, see :cpp:func:`make_min_aggregation`.

    Returns
    -------
    Aggregation
        The min aggregation.
    """
    return Aggregation.from_libcudf(move(make_min_aggregation[aggregation]()))


cpdef Aggregation max():
    """Create a max aggregation.

    For details, see :cpp:func:`make_max_aggregation`.

    Returns
    -------
    Aggregation
        The max aggregation.
    """
    return Aggregation.from_libcudf(move(make_max_aggregation[aggregation]()))


cpdef Aggregation ewma(float center_of_mass, ewm_history history):
    """Create a EWMA aggregation.

    For details, see :cpp:func:`make_ewma_aggregation`.

    Parameters
    ----------
    center_of_mass : float
        The decay in terms of the center of mass
    history : ewm_history
        Whether or not to treat the history as infinite.

    Returns
    -------
    Aggregation
        The EWMA aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_ewma_aggregation[aggregation](center_of_mass, history))
    )


cpdef Aggregation count(null_policy null_handling = null_policy.EXCLUDE):
    """Create a count aggregation.

    For details, see :cpp:func:`make_count_aggregation`.

    Parameters
    ----------
    null_handling : null_policy, default EXCLUDE
        Whether or not nulls should be included.

    Returns
    -------
    Aggregation
        The count aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_count_aggregation[aggregation](null_handling))
    )


cpdef Aggregation any():
    """Create an any aggregation.

    For details, see :cpp:func:`make_any_aggregation`.

    Returns
    -------
    Aggregation
        The any aggregation.
    """
    return Aggregation.from_libcudf(move(make_any_aggregation[aggregation]()))


cpdef Aggregation all():
    """Create an all aggregation.

    For details, see :cpp:func:`make_all_aggregation`.

    Returns
    -------
    Aggregation
        The all aggregation.
    """
    return Aggregation.from_libcudf(move(make_all_aggregation[aggregation]()))


cpdef Aggregation sum_of_squares():
    """Create a sum_of_squares aggregation.

    For details, see :cpp:func:`make_sum_of_squares_aggregation`.

    Returns
    -------
    Aggregation
        The sum_of_squares aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_sum_of_squares_aggregation[aggregation]())
    )


cpdef Aggregation mean():
    """Create a mean aggregation.

    For details, see :cpp:func:`make_mean_aggregation`.

    Returns
    -------
    Aggregation
        The mean aggregation.
    """
    return Aggregation.from_libcudf(move(make_mean_aggregation[aggregation]()))


cpdef Aggregation variance(size_type ddof=1):
    """Create a variance aggregation.

    For details, see :cpp:func:`make_variance_aggregation`.

    Parameters
    ----------
    ddof : int, default 1
        Delta degrees of freedom.

    Returns
    -------
    Aggregation
        The variance aggregation.
    """
    return Aggregation.from_libcudf(move(make_variance_aggregation[aggregation](ddof)))


cpdef Aggregation std(size_type ddof=1):
    """Create a std aggregation.

    For details, see :cpp:func:`make_std_aggregation`.

    Parameters
    ----------
    ddof : int, default 1
        Delta degrees of freedom. The default value is 1.

    Returns
    -------
    Aggregation
        The std aggregation.
    """
    return Aggregation.from_libcudf(move(make_std_aggregation[aggregation](ddof)))


cpdef Aggregation median():
    """Create a median aggregation.

    For details, see :cpp:func:`make_median_aggregation`.

    Returns
    -------
    Aggregation
        The median aggregation.
    """
    return Aggregation.from_libcudf(move(make_median_aggregation[aggregation]()))


cpdef Aggregation quantile(list quantiles, interpolation interp = interpolation.LINEAR):
    """Create a quantile aggregation.

    For details, see :cpp:func:`make_quantile_aggregation`.

    Parameters
    ----------
    quantiles : list
        List of quantiles to compute, should be between 0 and 1.
    interp : interpolation, default LINEAR
        Interpolation technique to use when the desired quantile lies between
        two data points.

    Returns
    -------
    Aggregation
        The quantile aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_quantile_aggregation[aggregation](quantiles, interp))
    )


cpdef Aggregation argmax():
    """Create an argmax aggregation.

    For details, see :cpp:func:`make_argmax_aggregation`.

    Returns
    -------
    Aggregation
        The argmax aggregation.
    """
    return Aggregation.from_libcudf(move(make_argmax_aggregation[aggregation]()))


cpdef Aggregation argmin():
    """Create an argmin aggregation.

    For details, see :cpp:func:`make_argmin_aggregation`.

    Returns
    -------
    Aggregation
        The argmin aggregation.
    """
    return Aggregation.from_libcudf(move(make_argmin_aggregation[aggregation]()))


cpdef Aggregation nunique(null_policy null_handling = null_policy.EXCLUDE):
    """Create a nunique aggregation.

    For details, see :cpp:func:`make_nunique_aggregation`.

    Parameters
    ----------
    null_handling : null_policy, default EXCLUDE
        Whether or not nulls should be included.

    Returns
    -------
    Aggregation
        The nunique aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_nunique_aggregation[aggregation](null_handling))
    )


cpdef Aggregation nth_element(
    size_type n, null_policy null_handling = null_policy.INCLUDE
):
    """Create a nth_element aggregation.

    For details, see :cpp:func:`make_nth_element_aggregation`.

    Parameters
    ----------
    null_handling : null_policy, default INCLUDE
        Whether or not nulls should be included.

    Returns
    -------
    Aggregation
        The nth_element aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_nth_element_aggregation[aggregation](n, null_handling))
    )


cpdef Aggregation collect_list(null_policy null_handling = null_policy.INCLUDE):
    """Create a collect_list aggregation.

    For details, see :cpp:func:`make_collect_list_aggregation`.

    Parameters
    ----------
    null_handling : null_policy, default INCLUDE
        Whether or not nulls should be included.

    Returns
    -------
    Aggregation
        The collect_list aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_collect_list_aggregation[aggregation](null_handling))
    )


cpdef Aggregation collect_set(
    null_handling = null_policy.INCLUDE,
    nulls_equal = null_equality.EQUAL,
    nans_equal = nan_equality.ALL_EQUAL,
):
    """Create a collect_set aggregation.

    For details, see :cpp:func:`make_collect_set_aggregation`.

    Parameters
    ----------
    null_handling : null_policy, default INCLUDE
        Whether or not nulls should be included.
    nulls_equal : null_equality, default EQUAL
        Whether or not nulls should be considered equal.
    nans_equal : nan_equality, default ALL_EQUAL
        Whether or not NaNs should be considered equal.

    Returns
    -------
    Aggregation
        The collect_set aggregation.
    """
    return Aggregation.from_libcudf(
        move(
            make_collect_set_aggregation[aggregation](
                null_handling, nulls_equal, nans_equal
            )
        )
    )

cpdef Aggregation udf(str operation, DataType output_type):
    """Create a udf aggregation.

    For details, see :cpp:func:`make_udf_aggregation`.

    Parameters
    ----------
    operation : str
        The operation to perform as a string of PTX code.
    output_type : DataType
        The output type of the aggregation.

    Returns
    -------
    Aggregation
        The udf aggregation.
    """
    return Aggregation.from_libcudf(
        move(
            make_udf_aggregation[aggregation](
                UdfType.PTX,
                operation.encode("utf-8"),
                output_type.c_obj,
            )
        )
    )


cpdef Aggregation correlation(correlation_type type, size_type min_periods):
    """Create a correlation aggregation.

    For details, see :cpp:func:`make_correlation_aggregation`.

    Parameters
    ----------
    type : correlation_type
        The type of correlation to compute.
    min_periods : int
        The minimum number of observations to consider for computing the
        correlation.

    Returns
    -------
    Aggregation
        The correlation aggregation.
    """
    return Aggregation.from_libcudf(
        move(make_correlation_aggregation[aggregation](type, min_periods))
    )


cpdef Aggregation covariance(size_type min_periods, size_type ddof):
    """Create a covariance aggregation.

    For details, see :cpp:func:`make_covariance_aggregation`.

    Parameters
    ----------
    min_periods : int
        The minimum number of observations to consider for computing the
        covariance.
    ddof : int
        Delta degrees of freedom.

    Returns
    -------
    Aggregation
        The covariance aggregation.
    """
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
    """Create a rank aggregation.

    For details, see :cpp:func:`make_rank_aggregation`.

    Parameters
    ----------
    method : rank_method
        The method to use for ranking.
    column_order : order, default ASCENDING
        The order in which to sort the column.
    null_handling : null_policy, default EXCLUDE
        Whether or not nulls should be included.
    null_precedence : null_order, default AFTER
        Whether nulls should come before or after non-nulls.
    percentage : rank_percentage, default NONE
        Whether or not ranks should be converted to percentages, and if so,
        the type of normalization to use.

    Returns
    -------
    Aggregation
        The rank aggregation.
    """
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
