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

import contextlib
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

with contextlib.suppress(ImportError):
    from .utils import (
        get_data,
        run_pandas,
    )


if TYPE_CHECKING:
    from .utils import RunConfig


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

    @staticmethod
    def q2(run_config: RunConfig) -> pd.DataFrame:
        """Query 2."""
        raise NotImplementedError

    @staticmethod
    def q3(run_config: RunConfig) -> pd.DataFrame:
        """Query 3."""
        raise NotImplementedError

    @staticmethod
    def q4(run_config: RunConfig) -> pd.DataFrame:
        """Query 4."""
        raise NotImplementedError

    @staticmethod
    def q5(run_config: RunConfig) -> pd.DataFrame:
        """Query 5."""
        raise NotImplementedError

    @staticmethod
    def q6(run_config: RunConfig) -> pd.DataFrame:
        """Query 6."""
        raise NotImplementedError

    @staticmethod
    def q7(run_config: RunConfig) -> pd.DataFrame:
        """Query 7."""
        raise NotImplementedError

    @staticmethod
    def q8(run_config: RunConfig) -> pd.DataFrame:
        """Query 8."""
        raise NotImplementedError

    @staticmethod
    def q9(run_config: RunConfig) -> pd.DataFrame:
        """Query 9."""
        raise NotImplementedError

    @staticmethod
    def q10(run_config: RunConfig) -> pd.DataFrame:
        """Query 10."""
        raise NotImplementedError

    @staticmethod
    def q11(run_config: RunConfig) -> pd.DataFrame:
        """Query 11."""
        raise NotImplementedError

    @staticmethod
    def q12(run_config: RunConfig) -> pd.DataFrame:
        """Query 12."""
        raise NotImplementedError

    @staticmethod
    def q13(run_config: RunConfig) -> pd.DataFrame:
        """Query 13."""
        raise NotImplementedError

    @staticmethod
    def q14(run_config: RunConfig) -> pd.DataFrame:
        """Query 14."""
        raise NotImplementedError

    @staticmethod
    def q15(run_config: RunConfig) -> pd.DataFrame:
        """Query 15."""
        raise NotImplementedError

    @staticmethod
    def q16(run_config: RunConfig) -> pd.DataFrame:
        """Query 16."""
        raise NotImplementedError

    @staticmethod
    def q17(run_config: RunConfig) -> pd.DataFrame:
        """Query 17."""
        raise NotImplementedError

    @staticmethod
    def q18(run_config: RunConfig) -> pd.DataFrame:
        """Query 18."""
        raise NotImplementedError

    @staticmethod
    def q19(run_config: RunConfig) -> pd.DataFrame:
        """Query 19."""
        raise NotImplementedError

    @staticmethod
    def q20(run_config: RunConfig) -> pd.DataFrame:
        """Query 20."""
        raise NotImplementedError

    @staticmethod
    def q21(run_config: RunConfig) -> pd.DataFrame:
        """Query 21."""
        raise NotImplementedError

    @staticmethod
    def q22(run_config: RunConfig) -> pd.DataFrame:
        """Query 22."""
        raise NotImplementedError


if __name__ == "__main__":
    run_pandas(PDSHQueries)
