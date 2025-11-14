# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import bisect
import itertools

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


@pytest.fixture(params=["ungrouped", "grouped"])
def with_groups(request):
    return request.param


@pytest.fixture
def groups(with_groups):
    if with_groups == "ungrouped":
        return []
    return [1, 1, 1, 2, 2, 2, 2, 3, 3, 5]


@pytest.fixture(
    params=[plc.types.Order.ASCENDING, plc.types.Order.DESCENDING],
    ids=["ascending", "descending"],
)
def sort_order(request):
    return request.param


@pytest.fixture
def orderby(with_groups, sort_order):
    if with_groups == "ungrouped":
        values = [-5, -2, 0, 10, 20, 20, 36, 42, 73, 102]
        if sort_order == plc.types.Order.DESCENDING:
            values = values[::-1]
    else:
        if sort_order == plc.types.Order.ASCENDING:
            values = [-5, -2, 0, -100, 2, 2, 4, 5, 7, 1]
        else:
            values = [0, -2, -5, 4, 3, 2, -100, 7, 5, 1]
    return values


@pytest.fixture
def values():
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture(
    params=[
        "bounded_closed_positive",
        "bounded_closed_negative",
        "bounded_open_positive",
        "bounded_open_negative",
        "current_row",
        "unbounded",
    ]
)
def preceding_endpoint(request):
    if request.param == "unbounded":
        return plc.rolling.Unbounded()
    elif request.param == "current_row":
        return plc.rolling.CurrentRow()
    elif request.param == "bounded_closed_positive":
        return plc.rolling.BoundedClosed(plc.Scalar.from_arrow(pa.scalar(10)))
    elif request.param == "bounded_closed_negative":
        return plc.rolling.BoundedClosed(plc.Scalar.from_arrow(pa.scalar(-10)))
    elif request.param == "bounded_open_positive":
        return plc.rolling.BoundedOpen(plc.Scalar.from_arrow(pa.scalar(10)))
    elif request.param == "bounded_open_negative":
        return plc.rolling.BoundedOpen(plc.Scalar.from_arrow(pa.scalar(-10)))


@pytest.fixture(
    params=[
        "bounded_closed_positive",
        "bounded_closed_negative",
        "bounded_open_positive",
        "bounded_open_negative",
        "current_row",
        "unbounded",
    ]
)
def following_endpoint(request):
    if request.param == "unbounded":
        return plc.rolling.Unbounded()
    elif request.param == "current_row":
        return plc.rolling.CurrentRow()
    elif request.param == "bounded_closed_positive":
        return plc.rolling.BoundedClosed(plc.Scalar.from_arrow(pa.scalar(2)))
    elif request.param == "bounded_closed_negative":
        return plc.rolling.BoundedClosed(plc.Scalar.from_arrow(pa.scalar(-2)))
    elif request.param == "bounded_open_positive":
        return plc.rolling.BoundedOpen(plc.Scalar.from_arrow(pa.scalar(2)))
    elif request.param == "bounded_open_negative":
        return plc.rolling.BoundedOpen(plc.Scalar.from_arrow(pa.scalar(-2)))


@pytest.fixture
def expect(
    groups,
    orderby,
    values,
    sort_order,
    preceding_endpoint,
    following_endpoint,
):
    result = []
    if len(groups) == 0:
        offsets = [0, len(values)]
        labels = [0] * len(values)
    else:
        offsets = [
            *filter(
                lambda i: i == 0 or groups[i] != groups[i - 1],
                range(len(groups)),
            ),
            len(groups),
        ]
        labels = list(
            itertools.chain(
                *(
                    itertools.repeat(i, offsets[i + 1] - offsets[i])
                    for i in range(len(offsets) - 1)
                )
            )
        )
    prec_bisect = (
        bisect.bisect_right
        if isinstance(preceding_endpoint, plc.rolling.BoundedOpen)
        else bisect.bisect_left
    )
    foll_bisect = (
        bisect.bisect_left
        if isinstance(following_endpoint, plc.rolling.BoundedOpen)
        else bisect.bisect_right
    )
    if isinstance(
        preceding_endpoint,
        (plc.rolling.BoundedClosed, plc.rolling.BoundedOpen),
    ):
        prec_delta = preceding_endpoint.delta.to_arrow().as_py()
    else:
        prec_delta = 0
    if isinstance(
        following_endpoint,
        (plc.rolling.BoundedClosed, plc.rolling.BoundedOpen),
    ):
        foll_delta = following_endpoint.delta.to_arrow().as_py()
    else:
        foll_delta = 0
    if sort_order == plc.types.Order.ASCENDING:

        def key(k):
            return k
    else:

        def key(k):
            return -k

        prec_delta *= -1
        foll_delta *= -1

    for i in range(len(values)):
        gbegin = offsets[labels[i]]
        gend = offsets[labels[i] + 1]

        start = (
            gbegin
            if isinstance(preceding_endpoint, plc.rolling.Unbounded)
            else prec_bisect(
                orderby, key(orderby[i] - prec_delta), gbegin, gend, key=key
            )
        )
        end = (
            gend
            if isinstance(following_endpoint, plc.rolling.Unbounded)
            else foll_bisect(
                orderby, key(orderby[i] + foll_delta), gbegin, gend, key=key
            )
        )

        selection = values[start:end]
        if len(selection) == 0:
            result.append(None)
        else:
            result.append(selection)
    return pa.array(result, type=pa.list_(pa.int64()))


def test_rolling_windows(
    groups,
    orderby,
    values,
    sort_order,
    preceding_endpoint,
    following_endpoint,
    expect,
):
    if len(groups) == 0:
        keys = plc.Table([])
    else:
        keys = plc.Table([plc.Column.from_arrow(pa.array(groups))])

    orderby = plc.Column.from_arrow(pa.array(orderby))
    values = plc.Column.from_arrow(pa.array(values))

    request = plc.rolling.RollingRequest(
        values, 1, plc.aggregation.collect_list()
    )
    (got,) = plc.rolling.grouped_range_rolling_window(
        keys,
        orderby,
        sort_order,
        plc.types.NullOrder.AFTER,
        preceding_endpoint,
        following_endpoint,
        [request],
    ).columns()
    assert_column_eq(expect, got)
