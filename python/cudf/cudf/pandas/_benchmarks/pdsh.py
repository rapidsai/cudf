# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""
Experimental PDS-H benchmarks.

Based on https://github.com/pola-rs/polars-benchmark.

WARNING: This is an experimental (and unofficial)
benchmark script. It is not intended for public use
and may be modified or removed at any time.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from cudf.pandas._benchmarks.utils import (
    get_data,
    run_pandas,
)

if TYPE_CHECKING:
    from cudf.pandas._benchmarks.utils import RunConfig


class PDSHQueries:
    """PDS-H query definitions."""

    name: str = "pdsh"

    @staticmethod
    def q0(run_config: RunConfig) -> pd.DataFrame:
        """Query 0."""
        return pd.DataFrame()

    @staticmethod
    def q1(run_config: RunConfig) -> pd.DataFrame:
        """Query 1."""
        line_item_ds = get_data(
            run_config.dataset_path, "lineitem", run_config.suffix
        )
        # TODO: Remove this workaround cuDF supports subtraction between float and decimal
        line_item_ds["l_quantity"] = line_item_ds["l_quantity"].astype(
            "float64"
        )
        line_item_ds["l_extendedprice"] = line_item_ds[
            "l_extendedprice"
        ].astype("float64")
        line_item_ds["l_discount"] = line_item_ds["l_discount"].astype(
            "float64"
        )
        line_item_ds["l_tax"] = line_item_ds["l_tax"].astype("float64")

        var1 = date(1998, 9, 2)

        filt = line_item_ds[line_item_ds["l_shipdate"] <= var1]

        # This is lenient towards pandas as normally an optimizer should decide
        # that this could be computed before the groupby aggregation.
        # Other implementations don't enjoy this benefit.
        filt["disc_price"] = filt.l_extendedprice * (1.0 - filt.l_discount)
        filt["charge"] = (
            filt.l_extendedprice * (1.0 - filt.l_discount) * (1.0 + filt.l_tax)
        )

        gb = filt.groupby(["l_returnflag", "l_linestatus"], as_index=False)
        agg = gb.agg(
            sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
            sum_base_price=pd.NamedAgg(
                column="l_extendedprice", aggfunc="sum"
            ),
            sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
            sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
            avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
            avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
            avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
            count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
        )

        return agg.sort_values(["l_returnflag", "l_linestatus"])


if __name__ == "__main__":
    run_pandas(PDSHQueries)
