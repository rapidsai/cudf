/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/groupby.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/sorting.hpp>
#include <cudf/reduction.hpp>
#include <cudf/unary.hpp>
#include <cudf/stream_compaction.hpp>

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/contains.hpp>


#include "utils.hpp"

/*
select
    nation,
    o_year,
    sum(amount) as sum_profit
from
    (
        select
            n_name as nation,
            extract(year from o_orderdate) as o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
        from
            part,
            supplier,
            lineitem,
            partsupp,
            orders,
            nation
        where
                s_suppkey = l_suppkey
          and ps_suppkey = l_suppkey
          and ps_partkey = l_partkey
          and p_partkey = l_partkey
          and o_orderkey = l_orderkey
          and s_nationkey = n_nationkey
          and p_name like '%green%'
    ) as profit
group by
    nation,
    o_year
order by
    nation,
    o_year desc;
*/

int main() {
    std::string dataset_dir = "/home/jayjeetc/tpch_sf1/";

    // 1. Read out the table from parquet files
    auto lineitem = read_parquet(dataset_dir + "lineitem/part-0.parquet");
    auto nation = read_parquet(dataset_dir + "nation/part-0.parquet");
    auto orders = read_parquet(dataset_dir + "orders/part-0.parquet");
    auto part = read_parquet(dataset_dir + "part/part-0.parquet");
    auto partsupp = read_parquet(dataset_dir + "partsupp/part-0.parquet");
    auto supplier = read_parquet(dataset_dir + "supplier/part-0.parquet");

    // 2. Filter the part table using `p_name like '%green%'`
    auto p_name = part->table().column(1);
    auto mask = cudf::strings::like(
        cudf::strings_column_view(p_name), cudf::string_scalar("%green%"));
    auto part_filtered = apply_mask(part, mask);
    
    // 3. Join the tables
    /*
    
    supplier and lineitem on s_suppkey and l_suppkey // done

    partsupp and lineitem on ps_suppkey and l_suppkey // done 
    partsupp and lineitem on ps_partkey and l_partkey // done

    part and lineitem on p_partkey and l_partkey

    orders and lineitem on o_orderkey and l_orderkey

    nation and supplier on n_nationkey and s_nationkey // done
    
    */
   std::cout << "Joining tables" << std::endl;
    auto join_a = apply_inner_join(
        lineitem,
        supplier,
        {"l_suppkey"}, 
        {"s_suppkey"}
    );
    auto join_b = apply_inner_join(
        join_a,
        partsupp,
        {"l_suppkey", "l_partkey"}, 
        {"ps_suppkey", "ps_partkey"}
    );
    auto join_c = apply_inner_join(
        join_b,
        part_filtered,
        {"l_partkey"}, 
        {"p_partkey"}
    );
    auto join_d = apply_inner_join(
        join_c,
        orders,
        {"l_orderkey"}, 
        {"o_orderkey"}
    );
    auto join_e = apply_inner_join(
        join_d,
        nation,
        {"s_nationkey"}, 
        {"n_nationkey"}
    );
}
