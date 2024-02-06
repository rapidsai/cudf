# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from enum import Enum, IntEnum

import pandas as pd

from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES, NullHandling
from cudf.utils import cudautils

from cudf._lib.types cimport (
    underlying_type_t_null_policy,
    underlying_type_t_type_id,
)

from numba.np import numpy_support

cimport cudf._lib.cpp.aggregation as libcudf_aggregation
cimport cudf._lib.cpp.types as libcudf_types
from cudf._lib.cpp.aggregation cimport underlying_type_t_correlation_type

import cudf

from cudf._lib cimport pylibcudf

from cudf._lib import pylibcudf


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
    RANK = libcudf_aggregation.aggregation.Kind.RANK
    COLLECT = libcudf_aggregation.aggregation.Kind.COLLECT_LIST
    UNIQUE = libcudf_aggregation.aggregation.Kind.COLLECT_SET
    PTX = libcudf_aggregation.aggregation.Kind.PTX
    CUDA = libcudf_aggregation.aggregation.Kind.CUDA
    CORRELATION = libcudf_aggregation.aggregation.Kind.CORRELATION
    COVARIANCE = libcudf_aggregation.aggregation.Kind.COVARIANCE


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


class RankMethod(IntEnum):
    FIRST = libcudf_aggregation.rank_method.FIRST
    AVERAGE = libcudf_aggregation.rank_method.AVERAGE
    MIN = libcudf_aggregation.rank_method.MIN
    MAX = libcudf_aggregation.rank_method.MAX
    DENSE = libcudf_aggregation.rank_method.DENSE


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
                rolling_aggregation](libcudf_types.null_policy.INCLUDE))
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

cdef class Aggregation:
    def __init__(self, pylibcudf.aggregation.Aggregation agg):
        self.c_obj = agg

    @property
    def kind(self):
        return AggregationKind(int(self.c_obj.kind())).name

    @classmethod
    def sum(cls):
        return cls(pylibcudf.aggregation.sum())

    @classmethod
    def min(cls):
        return cls(pylibcudf.aggregation.min())

    @classmethod
    def max(cls):
        return cls(pylibcudf.aggregation.max())

    @classmethod
    def idxmin(cls):
        return cls(pylibcudf.aggregation.argmin())

    @classmethod
    def idxmax(cls):
        return cls(pylibcudf.aggregation.argmax())

    @classmethod
    def mean(cls):
        return cls(pylibcudf.aggregation.mean())

    @classmethod
    def count(cls, dropna=True):
        return cls(pylibcudf.aggregation.count(
            pylibcudf.types.NullPolicy.EXCLUDE
            if dropna else pylibcudf.types.NullPolicy.INCLUDE
        ))

    @classmethod
    def size(cls):
        return cls(pylibcudf.aggregation.count(pylibcudf.types.NullPolicy.INCLUDE))

    @classmethod
    def collect(cls):
        return cls(
            pylibcudf.aggregation.collect_list(pylibcudf.types.NullPolicy.INCLUDE)
        )

    @classmethod
    def nunique(cls):
        return cls(pylibcudf.aggregation.nunique(pylibcudf.types.NullPolicy.EXCLUDE))

    @classmethod
    def nth(cls, libcudf_types.size_type size):
        return cls(pylibcudf.aggregation.nth_element(size))

    @classmethod
    def product(cls):
        return cls(pylibcudf.aggregation.product())
    prod = product

    @classmethod
    def sum_of_squares(cls):
        return cls(pylibcudf.aggregation.sum_of_squares())

    @classmethod
    def var(cls, ddof=1):
        return cls(pylibcudf.aggregation.variance(ddof))

    @classmethod
    def std(cls, ddof=1):
        return cls(pylibcudf.aggregation.std(ddof))

    @classmethod
    def median(cls):
        return cls(pylibcudf.aggregation.median())

    @classmethod
    def quantile(cls, q=0.5, interpolation="linear"):
        if not pd.api.types.is_list_like(q):
            q = [q]

        return cls(pylibcudf.aggregation.quantile(
            q, pylibcudf.types.Interpolation[interpolation.upper()]
        ))

    @classmethod
    def unique(cls):
        return cls(pylibcudf.aggregation.collect_set(
                pylibcudf.types.NullPolicy.INCLUDE,
                pylibcudf.types.NullEquality.EQUAL,
                pylibcudf.types.NanEquality.ALL_EQUAL,

        ))

    @classmethod
    def first(cls):
        return cls(
            pylibcudf.aggregation.nth_element(0, pylibcudf.types.NullPolicy.EXCLUDE)
        )

    @classmethod
    def last(cls):
        return cls(
            pylibcudf.aggregation.nth_element(-1, pylibcudf.types.NullPolicy.EXCLUDE)
        )

    @classmethod
    def corr(cls, method, libcudf_types.size_type min_periods):
        return cls(pylibcudf.aggregation.correlation(
            pylibcudf.aggregation.CorrelationType[method.upper()],
            min_periods

        ))

    @classmethod
    def cov(
        cls,
        libcudf_types.size_type min_periods,
        libcudf_types.size_type ddof=1
    ):
        return cls(pylibcudf.aggregation.covariance(
            min_periods,
            ddof
        ))

    # scan aggregations
    @classmethod
    def cumcount(cls):
        return cls.count(False)

    cumsum = sum
    cummin = min
    cummax = max

    @classmethod
    def rank(cls, method, ascending, na_option, pct):
        return cls(pylibcudf.aggregation.rank(
            pylibcudf.aggregation.RankMethod[method.upper()],
            (pylibcudf.types.Order.ASCENDING if ascending else
                pylibcudf.types.Order.DESCENDING),
            (pylibcudf.types.NullPolicy.EXCLUDE if na_option == "keep" else
                pylibcudf.types.NullPolicy.INCLUDE),
            (pylibcudf.types.NullOrder.BEFORE
                if (na_option == "top") == ascending else
                pylibcudf.types.NullOrder.AFTER),
            (pylibcudf.aggregation.RankPercentage.ZERO_NORMALIZED
                if pct else
                pylibcudf.aggregation.RankPercentage.NONE)

        ))

    # Reduce aggregations
    @classmethod
    def any(cls):
        return cls(pylibcudf.aggregation.any())

    @classmethod
    def all(cls):
        return cls(pylibcudf.aggregation.all())


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
