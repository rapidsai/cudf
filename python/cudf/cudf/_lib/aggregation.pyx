# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import pandas as pd
from numba.np import numpy_support

import cudf
from cudf._lib import pylibcudf
from cudf._lib.types import SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES
from cudf.utils import cudautils

_agg_name_map = {
    "COUNT_VALID": "COUNT",
    "COUNT_ALL": "SIZE",
    "VARIANCE": "VAR",
    "NTH_ELEMENT": "NTH",
    "COLLECT_LIST": "COLLECT",
    "COLLECT_SET": "UNIQUE",
}


class Aggregation:
    def __init__(self, agg):
        self.c_obj = agg

    @property
    def kind(self):
        name = self.c_obj.kind().name
        return _agg_name_map.get(name, name)

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
    def ewma(cls, com=1.0, adjust=True):
        return cls(pylibcudf.aggregation.ewma(
            com,
            pylibcudf.aggregation.EWMHistory.INFINITE
            if adjust else pylibcudf.aggregation.EWMHistory.FINITE
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
    def nth(cls, size):
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
    def corr(cls, method, min_periods):
        return cls(pylibcudf.aggregation.correlation(
            pylibcudf.aggregation.CorrelationType[method.upper()],
            min_periods

        ))

    @classmethod
    def cov(cls, min_periods, ddof=1):
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
    cumprod = product

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

    # Rolling aggregations
    @classmethod
    def from_udf(cls, op, *args, **kwargs):
        # Handling UDF type
        nb_type = numpy_support.from_dtype(kwargs['dtype'])
        type_signature = (nb_type[:],)
        ptx_code, output_dtype = cudautils.compile_udf(op, type_signature)
        output_np_dtype = cudf.dtype(output_dtype)
        if output_np_dtype not in SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES:
            raise TypeError(f"Result of window function has unsupported dtype {op[1]}")

        return cls(
            pylibcudf.aggregation.udf(
                ptx_code,
                pylibcudf.DataType(SUPPORTED_NUMPY_TO_PYLIBCUDF_TYPES[output_np_dtype]),
            )
        )


def make_aggregation(op, kwargs=None):
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

    if isinstance(op, str):
        return getattr(Aggregation, op)(**kwargs)
    elif callable(op):
        if op is list:
            return Aggregation.collect()
        elif "dtype" in kwargs:
            return Aggregation.from_udf(op, **kwargs)
        else:
            return op(Aggregation)
    raise TypeError(f"Unknown aggregation {op}")
