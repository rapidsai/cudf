from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from queries.pandas import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 8


def q() -> None:
    customer_ds = utils.get_customer_ds
    line_item_ds = utils.get_line_item_ds
    nation_ds = utils.get_nation_ds
    orders_ds = utils.get_orders_ds
    part_ds = utils.get_part_ds
    region_ds = utils.get_region_ds
    supplier_ds = utils.get_supplier_ds

    # first call one time to cache in case we don't include the IO times
    customer_ds()
    line_item_ds()
    nation_ds()
    orders_ds()
    part_ds()
    region_ds()
    supplier_ds()

    def query() -> pd.DataFrame:
        nonlocal customer_ds
        nonlocal line_item_ds
        nonlocal nation_ds
        nonlocal orders_ds
        nonlocal part_ds
        nonlocal region_ds
        nonlocal supplier_ds
        customer_ds = customer_ds()
        line_item_ds = line_item_ds()
        nation_ds = nation_ds()
        orders_ds = orders_ds()
        part_ds = part_ds()
        region_ds = region_ds()
        supplier_ds = supplier_ds()

        var1 = "BRAZIL"
        var2 = "AMERICA"
        var3 = "ECONOMY ANODIZED STEEL"
        var4 = date(1995, 1, 1)
        var5 = date(1996, 12, 31)

        n1 = nation_ds.loc[:, ["n_nationkey", "n_regionkey"]]
        n2 = nation_ds.loc[:, ["n_nationkey", "n_name"]]

        jn1 = part_ds.merge(line_item_ds, left_on="p_partkey", right_on="l_partkey")
        jn2 = jn1.merge(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        jn3 = jn2.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
        jn4 = jn3.merge(customer_ds, left_on="o_custkey", right_on="c_custkey")
        jn5 = jn4.merge(n1, left_on="c_nationkey", right_on="n_nationkey")
        jn6 = jn5.merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")

        jn6 = jn6[(jn6["r_name"] == var2)]

        jn7 = jn6.merge(n2, left_on="s_nationkey", right_on="n_nationkey")

        jn7 = jn7[(jn7["o_orderdate"] >= var4) & (jn7["o_orderdate"] <= var5)]
        jn7 = jn7[jn7["p_type"] == var3]

        jn7["o_year"] = jn7["o_orderdate"].dt.year
        jn7["volume"] = jn7["l_extendedprice"] * (1.0 - jn7["l_discount"])
        jn7 = jn7.rename(columns={"n_name": "nation"})

        def udf(df: pd.DataFrame) -> float:
            demonimator: float = df["volume"].sum()
            df = df[df["nation"] == var1]
            numerator: float = df["volume"].sum()
            return round(numerator / demonimator, 2)

        gb = jn7.groupby("o_year", as_index=False)
        agg = gb.apply(udf, include_groups=False)
        agg.columns = ["o_year", "mkt_share"]
        result_df = agg.sort_values("o_year")

        return result_df  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
