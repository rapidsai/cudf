from __future__ import annotations

from datetime import date

import pandas as pd

from queries.pandas import utils

Q_NUM = 7


def q() -> None:
    nation_ds = utils.get_nation_ds
    customer_ds = utils.get_customer_ds
    line_item_ds = utils.get_line_item_ds
    orders_ds = utils.get_orders_ds
    supplier_ds = utils.get_supplier_ds

    # first call one time to cache in case we don't include the IO times
    nation_ds()
    customer_ds()
    line_item_ds()
    orders_ds()
    supplier_ds()

    def query() -> pd.DataFrame:
        nonlocal nation_ds
        nonlocal customer_ds
        nonlocal line_item_ds
        nonlocal orders_ds
        nonlocal supplier_ds
        nation_ds = nation_ds()
        customer_ds = customer_ds()
        line_item_ds = line_item_ds()
        orders_ds = orders_ds()
        supplier_ds = supplier_ds()

        var1 = "FRANCE"
        var2 = "GERMANY"
        var3 = date(1995, 1, 1)
        var4 = date(1996, 12, 31)

        n1 = nation_ds[(nation_ds["n_name"] == var1)]
        n2 = nation_ds[(nation_ds["n_name"] == var2)]

        # Part 1
        jn1 = customer_ds.merge(n1, left_on="c_nationkey", right_on="n_nationkey")
        jn2 = jn1.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn2.rename(columns={"n_name": "cust_nation"})
        jn3 = jn2.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        jn4 = jn3.merge(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        jn5 = jn4.merge(n2, left_on="s_nationkey", right_on="n_nationkey")
        df1 = jn5.rename(columns={"n_name": "supp_nation"})

        # Part 2
        jn1 = customer_ds.merge(n2, left_on="c_nationkey", right_on="n_nationkey")
        jn2 = jn1.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn2.rename(columns={"n_name": "cust_nation"})
        jn3 = jn2.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        jn4 = jn3.merge(supplier_ds, left_on="l_suppkey", right_on="s_suppkey")
        jn5 = jn4.merge(n1, left_on="s_nationkey", right_on="n_nationkey")
        df2 = jn5.rename(columns={"n_name": "supp_nation"})

        # Combine
        total = pd.concat([df1, df2])

        total = total[(total["l_shipdate"] >= var3) & (total["l_shipdate"] <= var4)]
        total["volume"] = total["l_extendedprice"] * (1.0 - total["l_discount"])
        total["l_year"] = total["l_shipdate"].dt.year

        gb = total.groupby(["supp_nation", "cust_nation", "l_year"], as_index=False)
        agg = gb.agg(revenue=pd.NamedAgg(column="volume", aggfunc="sum"))

        result_df = agg.sort_values(by=["supp_nation", "cust_nation", "l_year"])

        return result_df  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
