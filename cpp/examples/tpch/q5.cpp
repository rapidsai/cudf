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
#include <cudf/io/parquet.hpp>
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

#include "utils.hpp"

/*
create view lineitem as select * from '~/tpch_sf1/lineitem/part-0.parquet';
create view orders as select * from '~/tpch_sf1/orders/part-0.parquet';
create view customer as select * from '~/tpch_sf1/customer/part-0.parquet';
create view supplier as select * from '~/tpch_sf1/supplier/part-0.parquet';
create view nation as select * from '~/tpch_sf1/nation/part-0.parquet';
create view region as select * from '~/tpch_sf1/region/part-0.parquet';

select
    n_name,
    sum(l_extendedprice * (1 - l_discount)) as revenue
from
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
where
        c_custkey = o_custkey
  and l_orderkey = o_orderkey
  and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey
  and s_nationkey = n_nationkey
  and n_regionkey = r_regionkey
  and r_name = 'ASIA'
  and o_orderdate >= date '1994-01-01'
  and o_orderdate < date '1995-01-01'
group by
    n_name
order by
    revenue desc;
*/

std::unique_ptr<cudf::column> calc_disc_price(std::unique_ptr<cudf::table>& table) {
    auto one = cudf::fixed_point_scalar<numeric::decimal64>(1, -2);
    auto disc = table->get_column(37).view();
    auto one_minus_disc = cudf::binary_operation(one, disc, cudf::binary_operator::SUB, disc.type());
    auto extended_price = table->get_column(36).view();

    auto disc_price_type = cudf::data_type{cudf::type_id::DECIMAL64, -4};
    auto disc_price = cudf::binary_operation(extended_price, one_minus_disc->view(), cudf::binary_operator::MUL, disc_price_type);
    return disc_price;
}

int main() {
    std::string customer_path = "/home/jayjeetc/tpch_sf1/customer/part-0.parquet";
    std::string orders_path = "/home/jayjeetc/tpch_sf1/orders/part-0.parquet";
    std::string lineitem_path = "/home/jayjeetc/tpch_sf1/lineitem/part-0.parquet";
    std::string supplier_path = "/home/jayjeetc/tpch_sf1/supplier/part-0.parquet";
    std::string nation_path = "/home/jayjeetc/tpch_sf1/nation/part-0.parquet";
    std::string region_path = "/home/jayjeetc/tpch_sf1/region/part-0.parquet";

    // read out the tables along with their column names
    auto customer = read_parquet(customer_path);
    auto orders = read_parquet(orders_path);
    auto lineitem = read_parquet(lineitem_path);
    auto supplier = read_parquet(supplier_path);
    auto nation = read_parquet(nation_path);
    auto region = read_parquet(region_path);

    // move the tables out of the pair
    auto customer_table = std::move(customer.first);
    auto orders_table = std::move(orders.first);
    auto lineitem_table = std::move(lineitem.first);
    auto supplier_table = std::move(supplier.first);
    auto nation_table = std::move(nation.first);
    auto region_table = std::move(region.first);

    // join_a: region with nation on r_regionkey = n_regionkey
    auto join_a = inner_join(region_table->view(), nation_table->view(), {0}, {2});
    auto join_a_column_names = concat(region.second, nation.second);

    // join_b: join_a with customer on n_nationkey = c_nationkey
    auto join_b = inner_join(join_a->view(), customer_table->view(), {3}, {3});
    auto join_b_column_names = concat(join_a_column_names, customer.second);

    // join_c: join_b with orders on c_custkey = o_custkey
    auto join_c = inner_join(join_b->view(), orders_table->view(), {7}, {1});
    auto join_c_column_names = concat(join_b_column_names, orders.second);

    // join_d: join_c with lineitem on o_orderkey = l_orderkey
    auto join_d = inner_join(join_c->view(), lineitem_table->view(), {15}, {0});
    auto join_d_column_names = concat(join_c_column_names, lineitem.second);

    // join_e: join_d with supplier on l_suppkey = s_suppkey
    auto join_e = inner_join(supplier_table->view(), join_d->view(), {0, 3}, {26, 3});
    auto join_e_column_names = concat(supplier.second, join_d_column_names);

    // apply filter predicates
    auto o_orderdate = cudf::ast::column_reference(26);
    
    auto o_orderdate_lower = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
    auto o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
    auto pred_a = cudf::ast::operation(
        cudf::ast::ast_operator::GREATER_EQUAL,
        o_orderdate,
        o_orderdate_lower_limit
    );
    
    auto o_orderdate_upper = cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
    auto o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
    auto pred_b = cudf::ast::operation(
        cudf::ast::ast_operator::LESS,
        o_orderdate,
        o_orderdate_upper_limit
    );

    auto r_name = cudf::ast::column_reference(8); 

    auto r_name_value = cudf::string_scalar("ASIA");
    auto r_name_literal = cudf::ast::literal(r_name_value);
    auto pred_c = cudf::ast::operation(
        cudf::ast::ast_operator::EQUAL,
        r_name,
        r_name_literal
    );

    auto pred_ab = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_a,
        pred_b
    );

    auto pred_abc = cudf::ast::operation(
        cudf::ast::ast_operator::LOGICAL_AND,
        pred_ab,
        pred_c
    );

    auto filtered_table = apply_filter(join_e, pred_abc);

    // calcute revenue column
    auto revenue_col = calc_disc_price(filtered_table);
    auto new_table = append_col_to_table(filtered_table, revenue_col);

    // perform group by
    groupby_context ctx{{11}, {{
        47, {cudf::aggregation::Kind::SUM}
    }}};
    auto groupedby_table = apply_groupby(new_table, ctx);
    auto orderedby_table = apply_orderby(groupedby_table, {1});
    write_parquet(orderedby_table, create_table_metadata({"n_name", "revenue"}), "q5.parquet");
}
