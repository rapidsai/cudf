# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 41."""

from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.pdsds_parameters import load_parameters
from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

Q_NUM = 41

MANUFACTURER_ID_START = 765


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 41."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=41,
        qualification=run_config.qualification,
    )

    manufact = params["manufact"]
    rules = params["rules"]

    return f"""
    SELECT Distinct(i_product_name)
    FROM   item i1
    WHERE  i_manufact_id BETWEEN {manufact} AND {manufact} + 40
        AND (SELECT Count(*) AS item_cnt
                FROM   item
                WHERE  ( i_manufact = i1.i_manufact
                        AND ( {
        " or\n        ".join(
            [
                f"(i_category = '{rules[0]['category']}' and (i_color = '{rules[0]['colors'][0]}' or i_color = '{rules[0]['colors'][1]}') and (i_units = '{rules[0]['units'][0]}' or i_units = '{rules[0]['units'][1]}') and (i_size = '{rules[0]['sizes'][0]}' or i_size = '{rules[0]['sizes'][1]}'))",
                f"(i_category = '{rules[1]['category']}' and (i_color = '{rules[1]['colors'][0]}' or i_color = '{rules[1]['colors'][1]}') and (i_units = '{rules[1]['units'][0]}' or i_units = '{rules[1]['units'][1]}') and (i_size = '{rules[1]['sizes'][0]}' or i_size = '{rules[1]['sizes'][1]}'))",
                f"(i_category = '{rules[2]['category']}' and (i_color = '{rules[2]['colors'][0]}' or i_color = '{rules[2]['colors'][1]}') and (i_units = '{rules[2]['units'][0]}' or i_units = '{rules[2]['units'][1]}') and (i_size = '{rules[2]['sizes'][0]}' or i_size = '{rules[2]['sizes'][1]}'))",
                f"(i_category = '{rules[3]['category']}' and (i_color = '{rules[3]['colors'][0]}' or i_color = '{rules[3]['colors'][1]}') and (i_units = '{rules[3]['units'][0]}' or i_units = '{rules[3]['units'][1]}') and (i_size = '{rules[3]['sizes'][0]}' or i_size = '{rules[3]['sizes'][1]}'))",
            ]
        )
    } ) )
                        OR ( i_manufact = i1.i_manufact
                            AND ( {
        " or\n        ".join(
            [
                f"(i_category = '{rules[4]['category']}' and (i_color = '{rules[4]['colors'][0]}' or i_color = '{rules[4]['colors'][1]}') and (i_units = '{rules[4]['units'][0]}' or i_units = '{rules[4]['units'][1]}') and (i_size = '{rules[4]['sizes'][0]}' or i_size = '{rules[4]['sizes'][1]}'))",
                f"(i_category = '{rules[5]['category']}' and (i_color = '{rules[5]['colors'][0]}' or i_color = '{rules[5]['colors'][1]}') and (i_units = '{rules[5]['units'][0]}' or i_units = '{rules[5]['units'][1]}') and (i_size = '{rules[5]['sizes'][0]}' or i_size = '{rules[5]['sizes'][1]}'))",
                f"(i_category = '{rules[6]['category']}' and (i_color = '{rules[6]['colors'][0]}' or i_color = '{rules[6]['colors'][1]}') and (i_units = '{rules[6]['units'][0]}' or i_units = '{rules[6]['units'][1]}') and (i_size = '{rules[6]['sizes'][0]}' or i_size = '{rules[6]['sizes'][1]}'))",
                f"(i_category = '{rules[7]['category']}' and (i_color = '{rules[7]['colors'][0]}' or i_color = '{rules[7]['colors'][1]}') and (i_units = '{rules[7]['units'][0]}' or i_units = '{rules[7]['units'][1]}') and (i_size = '{rules[7]['sizes'][0]}' or i_size = '{rules[7]['sizes'][1]}'))",
            ]
        )
    } ) )) > 0
    ORDER  BY i_product_name
    LIMIT 100;

    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 41."""
    params = load_parameters(
        int(run_config.scale_factor),
        query_id=41,
        qualification=run_config.qualification,
    )

    manufact = params["manufact"]
    rules = params["rules"]

    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    # Convert rules from dict format to tuple format and build expressions
    rule_exprs = [
        (
            (pl.col("i_category") == rule["category"])
            & pl.col("i_color").is_in(rule["colors"])
            & pl.col("i_units").is_in(rule["units"])
            & pl.col("i_size").is_in(rule["sizes"])
        )
        for rule in rules
    ]
    subquery_conditions = reduce(op.or_, rule_exprs)

    manufacturers_with_criteria = (
        item.filter(subquery_conditions).select("i_manufact").unique()
    )

    return (
        item.filter(pl.col("i_manufact_id").is_between(manufact, manufact + 40))
        .join(manufacturers_with_criteria, on="i_manufact", how="inner")
        .select("i_product_name")
        .unique()
        .sort("i_product_name")
        .limit(100)
    )
