# Copyright (c) 2024, NVIDIA CORPORATION.
import itertools
import pickle

import pytest

from pylibcudf import DataType
from pylibcudf.aggregation import (
    Aggregation,
    EWMHistory,
    all as agg_all,
    any as agg_any,
    argmax,
    argmin,
    collect_list,
    collect_set,
    correlation,
    count,
    covariance,
    ewma,
    max as agg_max,
    mean,
    median,
    min as agg_min,
    nth_element,
    nunique,
    product as product,
    quantile,
    rank,
    std,
    sum as agg_sum,
    sum_of_squares,
    udf,
    variance,
)
from pylibcudf.libcudf.aggregation import (
    correlation_type,
    rank_method,
    rank_percentage,
)
from pylibcudf.libcudf.types import (
    interpolation,
    nan_equality,
    null_equality,
    null_order,
    null_policy,
    order,
)
from pylibcudf.types import TypeId


@pytest.fixture(params=list(TypeId))
def dtype(request):
    tid = request.param
    if tid in {TypeId.DECIMAL32, TypeId.DECIMAL64, TypeId.DECIMAL128}:
        scale = 5
    else:
        scale = 0
    return DataType(tid, scale)


def test_datatype_reduce(dtype):
    (typ, (tid, scale)) = dtype.__reduce__()
    assert typ is DataType
    assert tid == dtype.id()
    assert scale == dtype.scale()


def test_datatype_pickle(dtype):
    assert dtype == pickle.loads(pickle.dumps(dtype))


null_handling_choices = [
    {"null_handling": null_policy.EXCLUDE},
    {"null_handling": null_policy.INCLUDE},
]
ddof_choices = [
    {"ddof": 1},
    {"ddof": 5},
]
interpolation_choices = [
    {"interp": interpolation.LINEAR},
    {"interp": interpolation.LOWER},
    {"interp": interpolation.HIGHER},
    {"interp": interpolation.MIDPOINT},
    {"interp": interpolation.NEAREST},
]
center_of_mass_choices = [
    {"center_of_mass": 1.0},
    {"center_of_mass": 12.34},
]
ewh_history_choices = [
    {"history": EWMHistory.FINITE},
    {"history": EWMHistory.INFINITE},
]
ewma_kwargs_choices = [
    d1 | d2
    for d1, d2 in itertools.product(
        *[center_of_mass_choices, ewh_history_choices]
    )
]
column_order_choices = [
    {"column_order": order.ASCENDING},
    {"column_order": order.DESCENDING},
]
null_precedence_choices = [
    {"null_precedence": null_order.AFTER},
    {"null_precedence": null_order.BEFORE},
]
percentage_choices = [
    {"percentage": rank_percentage.NONE},
    {"percentage": rank_percentage.ZERO_NORMALIZED},
    {"percentage": rank_percentage.ONE_NORMALIZED},
]
rank_method_choices = [
    [rank_method.FIRST],
    [rank_method.AVERAGE],
    [rank_method.MIN],
    [rank_method.MAX],
    [rank_method.DENSE],
]
rank_kwargs_choices = [
    d1 | d2 | d3 | d4
    for d1, d2, d3, d4 in itertools.product(
        *[
            column_order_choices,
            null_handling_choices,
            null_precedence_choices,
            percentage_choices,
        ]
    )
]
nulls_equal_choices = [
    {"nulls_equal": null_equality.EQUAL},
    {"nulls_equal": null_equality.UNEQUAL},
]
nans_equal_choices = [
    {"nans_equal": nan_equality.ALL_EQUAL},
    {"nans_equal": nan_equality.UNEQUAL},
]
collect_set_kwargs_choices = [
    d1 | d2 | d3
    for d1, d2, d3 in itertools.product(
        *[null_handling_choices, nulls_equal_choices, nans_equal_choices]
    )
]
# count_choices = itertools.product([count], [[]], [{}, *null_handling_choices])
# print(count_choices)


@pytest.fixture(
    params=[
        (agg_sum, [], {}),
        (agg_min, [], {}),
        (agg_max, [], {}),
        (product, [], {}),
        *itertools.product([count], [[]], [{}, *null_handling_choices]),
        (agg_any, [], {}),
        (agg_all, [], {}),
        (sum_of_squares, [], {}),
        (mean, [], {}),
        *itertools.product([variance], [[]], [{}, *ddof_choices]),
        *itertools.product([std], [[]], [{}, *ddof_choices]),
        (median, [], {}),
        *itertools.product(
            [quantile],
            [[[0.1, 0.9]], [[0.25, 0.5, 0.75]]],
            [{}, *interpolation_choices],
        ),
        (argmax, [], {}),
        (argmin, [], {}),
        *itertools.product([nunique], [[]], [{}, *null_handling_choices]),
        *itertools.product(
            [nth_element], [[0], [5]], [{}, *null_handling_choices]
        ),
        *itertools.product([ewma], [[]], ewma_kwargs_choices),
        *itertools.product([rank], rank_method_choices, rank_kwargs_choices),
        *itertools.product([collect_list], [[]], [{}, *null_handling_choices]),
        *itertools.product(
            [collect_set], [[]], [{}, *collect_set_kwargs_choices]
        ),
        *itertools.product(
            [udf],
            itertools.product(["x = 1"], [DataType(tid, 0) for tid in TypeId]),
            [{}],
        ),
        *itertools.product(
            [correlation],
            itertools.product(
                [
                    correlation_type.PEARSON,
                    correlation_type.KENDALL,
                    correlation_type.SPEARMAN,
                ],
                [2, 3],
            ),
            [{}],
        ),
        *itertools.product(
            [covariance], itertools.product([2, 3], [0, 5]), [{}]
        ),
    ]
)
def aggregation(request):
    function, args, kwargs = request.param
    return function(*args, **kwargs)


def test_aggregation_reduce(request, aggregation):
    (agg, args) = aggregation.__reduce__()
    assert type(aggregation) is Aggregation
    assert agg(*args).kind() == aggregation.kind()


def test_aggregation_pickle(aggregation):
    assert aggregation == pickle.loads(pickle.dumps(aggregation))
