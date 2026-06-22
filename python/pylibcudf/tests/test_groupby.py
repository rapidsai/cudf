# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest

import pylibcudf as plc


@pytest.mark.parametrize(
    "agg",
    [plc.aggregation.any, plc.aggregation.all],
)
def test_groupby_unsupported_bool_aggs_raise(agg):
    pa_table = pa.table(
        {
            "foo": pa.array(["a", "a", "b", "b", "c"]),
            "bar": pa.array([True, False, True, True, False]),
        }
    )
    plc_tbl = plc.Table.from_arrow(pa_table)

    groupby = plc.groupby.GroupBy(plc_tbl)
    request = plc.groupby.GroupByRequest(plc_tbl.columns()[0], [agg()])

    with pytest.raises(
        NotImplementedError,
        match=r"Aggregation(.*) aggregations are not supported by groupby",
    ):
        groupby.aggregate([request])
