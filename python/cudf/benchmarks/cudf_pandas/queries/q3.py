from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from queries.pandas import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 3


def q() -> None:
    customer_ds = utils.get_customer_ds
    line_item_ds = utils.get_line_item_ds
    orders_ds = utils.get_orders_ds

    # first call one time to cache in case we don't include the IO times
    customer_ds()
    line_item_ds()
    orders_ds()

    def query() -> pd.DataFrame:
        nonlocal customer_ds
        nonlocal line_item_ds
        nonlocal orders_ds
        customer_ds = customer_ds()
        line_item_ds = line_item_ds()
        orders_ds = orders_ds()

        var1 = "BUILDING"
        var2 = date(1995, 3, 15)

        fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]

        jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
        jn2 = jn1.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")

        jn2 = jn2[jn2["o_orderdate"] < var2]
        jn2 = jn2[jn2["l_shipdate"] > var2]
        jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

        gb = jn2.groupby(
            ["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False
        )
        agg = gb["revenue"].sum()

        sel = agg.loc[:, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
        sel = sel.rename(columns={"o_orderkey": "l_orderkey"})

        sorted = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])
        result_df = sorted.head(10)

        return result_df  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
