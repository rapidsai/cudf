from __future__ import annotations

from typing import TYPE_CHECKING

from queries.pandas import utils

if TYPE_CHECKING:
    import pandas as pd

Q_NUM = 2


def q() -> None:
    region_ds = utils.get_region_ds
    nation_ds = utils.get_nation_ds
    supplier_ds = utils.get_supplier_ds
    part_ds = utils.get_part_ds
    part_supp_ds = utils.get_part_supp_ds

    # first call one time to cache in case we don't include the IO times
    region_ds()
    nation_ds()
    supplier_ds()
    part_ds()
    part_supp_ds()

    def query() -> pd.DataFrame:
        nonlocal region_ds
        nonlocal nation_ds
        nonlocal supplier_ds
        nonlocal part_ds
        nonlocal part_supp_ds
        region_ds = region_ds()
        nation_ds = nation_ds()
        supplier_ds = supplier_ds()
        part_ds = part_ds()
        part_supp_ds = part_supp_ds()

        var1 = 15
        var2 = "BRASS"
        var3 = "EUROPE"

        jn = (
            part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
            .merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
            .merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
            .merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")
        )

        jn = jn[jn["p_size"] == var1]
        jn = jn[jn["p_type"].str.endswith(var2)]
        jn = jn[jn["r_name"] == var3]

        gb = jn.groupby("p_partkey", as_index=False)
        agg = gb["ps_supplycost"].min()
        jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

        sel = jn2.loc[
            :,
            [
                "s_acctbal",
                "s_name",
                "n_name",
                "p_partkey",
                "p_mfgr",
                "s_address",
                "s_phone",
                "s_comment",
            ],
        ]

        sort = sel.sort_values(
            by=["s_acctbal", "n_name", "s_name", "p_partkey"],
            ascending=[False, True, True, True],
        )
        result_df = sort.head(100)

        return result_df  # type: ignore[no-any-return]

    utils.run_query(Q_NUM, query)


if __name__ == "__main__":
    q()
