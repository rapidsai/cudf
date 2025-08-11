# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Query 8."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from cudf_polars.experimental.benchmarks.utils import get_data

if TYPE_CHECKING:
    from cudf_polars.experimental.benchmarks.utils import RunConfig

"""
warning!, one filter removed to prevent zero row results

note: alternate zip code
  '70069',  # 93 preferred customers
  '60069',  # 87 preferred customers
  '78877',  # 87 preferred customers
  '60169',  # 87 preferred customers
  '68252',  # 86 preferred customers
  '71087',  # 84 preferred customers
  '71711',  # 84 preferred customers
  '68877',  # 84 preferred customers
  '55709',  # 82 preferred customers
"""

TARGET_YEAR = 1996
TARGET_QUARTER = 2
TARGET_ZIPS = [
    "67436",
    "26121",
    "38443",
    "63157",
    "68856",
    "19485",
    "86425",
    "26741",
    "70991",
    "60899",
    "63573",
    "47556",
    "56193",
    "93314",
    "87827",
    "62017",
    "85067",
    "95390",
    "48091",
    "10261",
    "81845",
    "41790",
    "42853",
    "24675",
    "12840",
    "60065",
    "84430",
    "57451",
    "24021",
    "91735",
    "75335",
    "71935",
    "34482",
    "56943",
    "70695",
    "52147",
    "56251",
    "28411",
    "86653",
    "23005",
    "22478",
    "29031",
    "34398",
    "15365",
    "42460",
    "33337",
    "59433",
    "73943",
    "72477",
    "74081",
    "74430",
    "64605",
    "39006",
    "11226",
    "49057",
    "97308",
    "42663",
    "18187",
    "19768",
    "43454",
    "32147",
    "76637",
    "51975",
    "11181",
    "45630",
    "33129",
    "45995",
    "64386",
    "55522",
    "26697",
    "20963",
    "35154",
    "64587",
    "49752",
    "66386",
    "30586",
    "59286",
    "13177",
    "66646",
    "84195",
    "74316",
    "36853",
    "32927",
    "12469",
    "11904",
    "36269",
    "17724",
    "55346",
    "12595",
    "53988",
    "65439",
    "28015",
    "63268",
    "73590",
    "29216",
    "82575",
    "69267",
    "13805",
    "91678",
    "79460",
    "94152",
    "14961",
    "15419",
    "48277",
    "62588",
    "55493",
    "28360",
    "14152",
    "55225",
    "18007",
    "53705",
    "56573",
    "80245",
    "71769",
    "57348",
    "36845",
    "13039",
    "17270",
    "22363",
    "83474",
    "25294",
    "43269",
    "77666",
    "15488",
    "99146",
    "64441",
    "43338",
    "38736",
    "62754",
    "48556",
    "86057",
    "23090",
    "38114",
    "66061",
    "18910",
    "84385",
    "23600",
    "19975",
    "27883",
    "65719",
    "19933",
    "32085",
    "49731",
    "40473",
    "27190",
    "46192",
    "23949",
    "44738",
    "12436",
    "64794",
    "68741",
    "15333",
    "24282",
    "49085",
    "31844",
    "71156",
    "48441",
    "17100",
    "98207",
    "44982",
    "20277",
    "71496",
    "96299",
    "37583",
    "22206",
    "89174",
    "30589",
    "61924",
    "53079",
    "10976",
    "13104",
    "42794",
    "54772",
    "15809",
    "56434",
    "39975",
    "13874",
    "30753",
    "77598",
    "78229",
    "59478",
    "12345",
    "55547",
    "57422",
    "42600",
    "79444",
    "29074",
    "29752",
    "21676",
    "32096",
    "43044",
    "39383",
    "37296",
    "36295",
    "63077",
    "16572",
    "31275",
    "18701",
    "40197",
    "48242",
    "27219",
    "49865",
    "84175",
    "30446",
    "25165",
    "13807",
    "72142",
    "70499",
    "70464",
    "71429",
    "18111",
    "70857",
    "29545",
    "36425",
    "52706",
    "36194",
    "42963",
    "75068",
    "47921",
    "74763",
    "90990",
    "89456",
    "62073",
    "88397",
    "73963",
    "75885",
    "62657",
    "12530",
    "81146",
    "57434",
    "25099",
    "41429",
    "98441",
    "48713",
    "52552",
    "31667",
    "14072",
    "13903",
    "44709",
    "85429",
    "58017",
    "38295",
    "44875",
    "73541",
    "30091",
    "12707",
    "23762",
    "62258",
    "33247",
    "78722",
    "77431",
    "14510",
    "35656",
    "72428",
    "92082",
    "35267",
    "43759",
    "24354",
    "90952",
    "11512",
    "21242",
    "22579",
    "56114",
    "32339",
    "52282",
    "41791",
    "24484",
    "95020",
    "28408",
    "99710",
    "11899",
    "43344",
    "72915",
    "27644",
    "62708",
    "74479",
    "17177",
    "32619",
    "12351",
    "91339",
    "31169",
    "57081",
    "53522",
    "16712",
    "34419",
    "71779",
    "44187",
    "46206",
    "96099",
    "61910",
    "53664",
    "12295",
    "31837",
    "33096",
    "10813",
    "63048",
    "31732",
    "79118",
    "73084",
    "72783",
    "84952",
    "46965",
    "77956",
    "39815",
    "32311",
    "75329",
    "48156",
    "30826",
    "49661",
    "13736",
    "92076",
    "74865",
    "88149",
    "92397",
    "52777",
    "68453",
    "32012",
    "21222",
    "52721",
    "24626",
    "18210",
    "42177",
    "91791",
    "75251",
    "82075",
    "44372",
    "45542",
    "20609",
    "60115",
    "17362",
    "22750",
    "90434",
    "31852",
    "54071",
    "33762",
    "14705",
    "40718",
    "56433",
    "30996",
    "40657",
    "49056",
    "23585",
    "66455",
    "41021",
    "74736",
    "72151",
    "37007",
    "21729",
    "60177",
    "84558",
    "59027",
    "93855",
    "60022",
    "86443",
    "19541",
    "86886",
    "30532",
    "39062",
    "48532",
    "34713",
    "52077",
    "22564",
    "64638",
    "15273",
    "31677",
    "36138",
    "62367",
    "60261",
    "80213",
    "42818",
    "25113",
    "72378",
    "69802",
    "69096",
    "55443",
    "28820",
    "13848",
    "78258",
    "37490",
    "30556",
    "77380",
    "28447",
    "44550",
    "26791",
    "70609",
    "82182",
    "33306",
    "43224",
    "22322",
    "86959",
    "68519",
    "14308",
    "46501",
    "81131",
    "34056",
    "61991",
    "19896",
    "87804",
    "65774",
    "92564",
]


def duckdb_impl(run_config: RunConfig) -> str:
    """Query 8."""
    return f"""
    -- start query 8 in stream 0 using template query8.tpl
    SELECT s_store_name,
                Sum(ss_net_profit)
    FROM   store_sales,
        date_dim,
        store,
        (SELECT ca_zip
            FROM   (SELECT Substr(ca_zip, 1, 5) ca_zip
                    FROM   customer_address
                    WHERE  Substr(ca_zip, 1, 5) IN ({", ".join(f"'{zip}'" for zip in TARGET_ZIPS)})
                    INTERSECT
                    SELECT ca_zip
                    FROM   (SELECT Substr(ca_zip, 1, 5) ca_zip,
                                Count(*)             cnt
                            FROM   customer_address,
                                customer
                            WHERE  ca_address_sk = c_current_addr_sk
                                AND c_preferred_cust_flag = 'Y'
                            GROUP  BY ca_zip
                            HAVING Count(*) > 10)A1)A2) V1
    WHERE  ss_store_sk = s_store_sk
        AND ss_sold_date_sk = d_date_sk
        AND d_qoy = {TARGET_QUARTER}
        AND d_year = {TARGET_YEAR}
        AND ( Substr(s_zip, 1, 2) = Substr(V1.ca_zip, 1, 2) )
    GROUP  BY s_store_name
    ORDER  BY s_store_name
    LIMIT 100;

    """


def polars_impl(run_config: RunConfig) -> pl.LazyFrame:
    """Query 8."""
    # Load required tables
    store_sales = get_data(run_config.dataset_path, "store_sales", run_config.suffix)
    date_dim = get_data(run_config.dataset_path, "date_dim", run_config.suffix)
    store = get_data(run_config.dataset_path, "store", run_config.suffix)
    customer_address = get_data(
        run_config.dataset_path, "customer_address", run_config.suffix
    )
    customer = get_data(run_config.dataset_path, "customer", run_config.suffix)

    # First subquery: get first 5 chars of zip codes from target list
    target_zips_5char = (
        customer_address.select(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .filter(pl.col("ca_zip").is_in(TARGET_ZIPS))
        .unique()
    )

    # Second subquery: preferred customers by zip with count > 10
    preferred_customer_zips = (
        customer_address.join(
            customer, left_on="ca_address_sk", right_on="c_current_addr_sk"
        )
        .filter(pl.col("c_preferred_cust_flag") == "Y")
        .group_by(pl.col("ca_zip").str.slice(0, 5).alias("ca_zip"))
        .agg(pl.len().alias("cnt"))
        .filter(pl.col("cnt") > 10)
        .select("ca_zip")
    )

    # INTERSECT: Get common zip codes between target list and preferred customer zips
    intersect_zips = target_zips_5char.join(
        preferred_customer_zips, on="ca_zip", how="inner"
    ).select("ca_zip")

    # Main query: join store_sales with date_dim, store, and filter by zip codes
    return (
        store_sales.join(date_dim, left_on="ss_sold_date_sk", right_on="d_date_sk")
        .join(store, left_on="ss_store_sk", right_on="s_store_sk")
        .join(
            intersect_zips,
            left_on=pl.col("s_zip").str.slice(0, 2),
            right_on=pl.col("ca_zip").str.slice(0, 2),
        )
        .filter(pl.col("d_qoy") == TARGET_QUARTER)
        .filter(pl.col("d_year") == TARGET_YEAR)
        .group_by("s_store_name")
        .agg(pl.col("ss_net_profit").sum().alias("sum"))
        .sort("s_store_name", nulls_last=True)
        .limit(100)
        .select([pl.col("s_store_name"), pl.col("sum").alias("sum(ss_net_profit)")])
    )
