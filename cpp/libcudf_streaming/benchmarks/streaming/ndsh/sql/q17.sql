select
    round(sum(l_extendedprice) / 7.0, 2) as avg_yearly
from
    lineitem,
    part
where
    p_partkey = l_partkey
    and p_brand = 'Brand#23'
    and p_container = 'MED BOX'
    and l_quantity < (
        select
            0.2 * avg(l_quantity)
        from
            read_parquet('/datasets/toaugspurger/tpch-rs/scale-10/lineitem/*.parquet') as lineitem
        where
            l_partkey = p_partkey
    )
