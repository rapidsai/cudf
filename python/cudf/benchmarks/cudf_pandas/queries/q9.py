from __future__ import annotations

from datetime import date

import pandas as pd

from queries.pandas import utils

Q_NUM = 9


def q() -> None:
    line_item_ds = utils.get_line_item_ds
    orders_ds = utils.get_orders_ds
    part_ds = utils.get_part_ds
    nation_ds = utils.get_nation_ds
    part_supp_ds = utils.get_part_supp_ds
    supplier_ds = utils.get_supplier_ds

    # first call one time to cache in case we don't include the IO times
    line_item_ds()
    orders_ds()
    part_ds()
    nation_ds()
    part_supp_ds()
    supplier_ds()

    def query() -> pd.DataFrame:
        nonlocal line_item_ds
        nonlocal orders_ds
        nonlocal part_ds
        nonlocal nation_ds
        nonlocal part_supp_ds
        nonlocal supplier_ds
        line_item_ds = line_item_ds()
        orders_ds = orders_ds()
        part_ds = part_ds()
        nation_ds = nation_ds()
        part_supp_ds = part_supp_ds()
        supplier_ds = supplier_ds()

        jn1 = part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        jn2 = jn1.merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        jn3 = jn2.merge(line_item_ds, left_on=["p_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
        jn4 = jn3.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        jn5 = jn4.merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")

        jn5 = jn5[jn5["p_name"].str.contains("green", regex=False)]

        jn5["o_year"] = jn5["o_orderdate"].dt.year
        jn5["amount"] = jn5["l_extendedprice"] * (1.0 - jn5["l_discount"]) - (jn5["ps_supplycost"] * jn5["l_quantity"])
        jn5 = jn5.rename(columns={"n_name": "nation"})

        gb = jn5.groupby(["nation", "o_year"], as_index=False, sort=False)
        agg = gb.agg(sum_profit=pd.NamedAgg(column="amount", aggfunc="sum"))
        sorted = agg.sort_values(by=["nation", "o_year"], ascending=[True, False])
        result_df = sorted.reset_index(drop=True)

        return result_df

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()