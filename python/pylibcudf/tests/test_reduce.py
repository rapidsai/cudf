# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "agg,dt",
    [
        (plc.aggregation.any(), pa.int8()),
        (plc.aggregation.all(), pa.uint8()),
        (plc.aggregation.sum(), pa.int16()),
        (plc.aggregation.product(), pa.uint16()),
        (plc.aggregation.min(), pa.string()),
        (plc.aggregation.max(), pa.timestamp("s")),
        (plc.aggregation.argmin(), pa.int8()),
        (plc.aggregation.argmax(), pa.uint8()),
        (plc.aggregation.mean(), pa.int32()),
        (plc.aggregation.sum_of_squares(), pa.uint32()),
        (plc.aggregation.std(), pa.float32()),
        (plc.aggregation.variance(), pa.float64()),
        (plc.aggregation.median(), pa.uint64()),
        (plc.aggregation.quantile([10]), pa.int64()),
        (plc.aggregation.count(), pa.duration("ms")),
        (plc.aggregation.nunique(), pa.decimal128(6, 1)),
        (plc.aggregation.nth_element(10), pa.float32()),
    ],
)
def test_is_valid_aggregation(agg, dt):
    plc_type = plc.DataType.from_arrow(dt)
    assert plc.reduce.is_valid_reduce_aggregation(plc_type, agg)


@pytest.mark.parametrize(
    "agg,dt",
    [
        (plc.aggregation.rank(0), pa.string()),
        (plc.aggregation.covariance(1, 1), pa.float32()),
        (plc.aggregation.correlation(0, 1), pa.float64()),
        (plc.aggregation.sum(), pa.string()),
        (plc.aggregation.sum(), pa.timestamp("ms")),
        (plc.aggregation.mean(), pa.decimal128(6, 1)),
    ],
)
def test_is_not_valid_aggregation(agg, dt):
    plc_type = plc.DataType.from_arrow(dt)
    assert not plc.reduce.is_valid_reduce_aggregation(plc_type, agg)
