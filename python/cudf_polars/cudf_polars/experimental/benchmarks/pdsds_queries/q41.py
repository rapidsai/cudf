# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 41."""

from __future__ import annotations

import operator as op
from functools import reduce
from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

Q_NUM = 41

MANUFACTURER_ID_START = 765


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 41."""
    return f"""
    SELECT Distinct(i_product_name)
    FROM   item i1
    WHERE  i_manufact_id BETWEEN {MANUFACTURER_ID_START} AND {MANUFACTURER_ID_START} + 40
        AND (SELECT Count(*) AS item_cnt
                FROM   item
                WHERE  ( i_manufact = i1.i_manufact
                        AND ( ( i_category = 'Women'
                                AND ( i_color = 'dim'
                                        OR i_color = 'green' )
                                AND ( i_units = 'Gross'
                                        OR i_units = 'Dozen' )
                                AND ( i_size = 'economy'
                                        OR i_size = 'petite' ) )
                                OR ( i_category = 'Women'
                                    AND ( i_color = 'navajo'
                                            OR i_color = 'aquamarine' )
                                    AND ( i_units = 'Case'
                                            OR i_units = 'Unknown' )
                                    AND ( i_size = 'large'
                                            OR i_size = 'N/A' ) )
                                OR ( i_category = 'Men'
                                    AND ( i_color = 'indian'
                                            OR i_color = 'dark' )
                                    AND ( i_units = 'Oz'
                                            OR i_units = 'Lb' )
                                    AND ( i_size = 'extra large'
                                            OR i_size = 'small' ) )
                                OR ( i_category = 'Men'
                                    AND ( i_color = 'peach'
                                            OR i_color = 'purple' )
                                    AND ( i_units = 'Tbl'
                                            OR i_units = 'Bunch' )
                                    AND ( i_size = 'economy'
                                            OR i_size = 'petite' ) ) ) )
                        OR ( i_manufact = i1.i_manufact
                            AND ( ( i_category = 'Women'
                                    AND ( i_color = 'orchid'
                                            OR i_color = 'peru' )
                                    AND ( i_units = 'Carton'
                                            OR i_units = 'Cup' )
                                    AND ( i_size = 'economy'
                                            OR i_size = 'petite' ) )
                                    OR ( i_category = 'Women'
                                        AND ( i_color = 'violet'
                                                OR i_color = 'papaya' )
                                        AND ( i_units = 'Ounce'
                                                OR i_units = 'Box' )
                                        AND ( i_size = 'large'
                                                OR i_size = 'N/A' ) )
                                    OR ( i_category = 'Men'
                                        AND ( i_color = 'drab'
                                                OR i_color = 'grey' )
                                        AND ( i_units = 'Each'
                                                OR i_units = 'N/A' )
                                        AND ( i_size = 'extra large'
                                                OR i_size = 'small' ) )
                                    OR ( i_category = 'Men'
                                        AND ( i_color = 'chocolate'
                                                OR i_color = 'antique' )
                                        AND ( i_units = 'Dram'
                                                OR i_units = 'Gram' )
                                        AND ( i_size = 'economy'
                                                OR i_size = 'petite' ) ) ) )) > 0
    ORDER  BY i_product_name
    LIMIT 100;

    """


rules: list[tuple[str, list[str], list[str], list[str]]] = [
    ("Women", ["dim", "green"], ["Gross", "Dozen"], ["economy", "petite"]),
    ("Women", ["navajo", "aquamarine"], ["Case", "Unknown"], ["large", "N/A"]),
    ("Men", ["indian", "dark"], ["Oz", "Lb"], ["extra large", "small"]),
    ("Men", ["peach", "purple"], ["Tbl", "Bunch"], ["economy", "petite"]),
    ("Women", ["orchid", "peru"], ["Carton", "Cup"], ["economy", "petite"]),
    ("Women", ["violet", "papaya"], ["Ounce", "Box"], ["large", "N/A"]),
    ("Men", ["drab", "grey"], ["Each", "N/A"], ["extra large", "small"]),
    ("Men", ["chocolate", "antique"], ["Dram", "Gram"], ["economy", "petite"]),
]


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 41."""
    item = get_data(run_config.dataset_path, "item", run_config.suffix)

    rule_exprs = [
        (
            (pl.col("i_category") == cat)
            & pl.col("i_color").is_in(colors)
            & pl.col("i_units").is_in(units)
            & pl.col("i_size").is_in(sizes)
        )
        for (cat, colors, units, sizes) in rules
    ]
    subquery_conditions = reduce(op.or_, rule_exprs)

    manufacturers_with_criteria = (
        item.filter(subquery_conditions).select("i_manufact").unique()
    )

    return (
        item.filter(
            pl.col("i_manufact_id").is_between(
                MANUFACTURER_ID_START, MANUFACTURER_ID_START + 40
            )
        )
        .join(manufacturers_with_criteria, on="i_manufact", how="inner")
        .select("i_product_name")
        .unique()
        .sort("i_product_name")
        .limit(100)
    )
