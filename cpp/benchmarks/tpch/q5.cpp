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

#include "utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

/**
 * @file q5.cpp
 * @brief Implement query 5 of the TPC-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 * create view region as select * from '/tables/scale-1/region.parquet';
 *
 * select
 *    n_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    supplier,
 *    nation,
 *    region
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and l_suppkey = s_suppkey
 *     and c_nationkey = s_nationkey
 *     and s_nationkey = n_nationkey
 *     and n_regionkey = r_regionkey
 *     and r_name = 'ASIA'
 *     and o_orderdate >= date '1994-01-01'
 *     and o_orderdate < date '1995-01-01'
 * group by
 *    n_name
 * order by
 *    revenue desc;
 */

/**
 * @brief Calculate the revenue column
 *
 * @param extendedprice The extended price column
 * @param discount The discount column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_revenue(
  cudf::column_view const& extendedprice,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const revenue_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto revenue            = cudf::binary_operation(extendedprice,
                                        one_minus_discount->view(),
                                        cudf::binary_operator::MUL,
                                        revenue_type,
                                        stream,
                                        mr);
  return revenue;
}

int main(int argc, char const** argv)
{
  auto const args = parse_args(argc, argv);

  // Use a memory pool
  auto resource = create_memory_resource(args.memory_resource_type);
  rmm::mr::set_current_device_resource(resource.get());

  // Define the column projection and filter predicate for the `orders` table
  std::vector<std::string> const orders_cols = {"o_custkey", "o_orderkey", "o_orderdate"};
  auto const o_orderdate_ref                 = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  // Define the column projection and filter predicate for the `region` table
  std::vector<std::string> const region_cols = {"r_regionkey", "r_name"};
  auto const r_name_ref                      = cudf::ast::column_reference(std::distance(
    region_cols.begin(), std::find(region_cols.begin(), region_cols.end(), "r_name")));
  auto r_name_value                          = cudf::string_scalar("ASIA");
  auto const r_name_literal                  = cudf::ast::literal(r_name_value);
  auto const region_pred                     = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::EQUAL, r_name_ref, r_name_literal);

  // Read out the tables from parquet files
  // while pushing down the column projections and filter predicates
  auto const customer =
    read_parquet(args.dataset_dir + "/customer.parquet", {"c_custkey", "c_nationkey"});
  auto const orders =
    read_parquet(args.dataset_dir + "/orders.parquet", orders_cols, std::move(orders_pred));
  auto const lineitem = read_parquet(args.dataset_dir + "/lineitem.parquet",
                                     {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"});
  auto const supplier =
    read_parquet(args.dataset_dir + "/supplier.parquet", {"s_suppkey", "s_nationkey"});
  auto const nation =
    read_parquet(args.dataset_dir + "/nation.parquet", {"n_nationkey", "n_regionkey", "n_name"});
  auto const region =
    read_parquet(args.dataset_dir + "/region.parquet", region_cols, std::move(region_pred));

  // Perform the joins
  auto const join_a = apply_inner_join(region, nation, {"r_regionkey"}, {"n_regionkey"});
  auto const join_b = apply_inner_join(join_a, customer, {"n_nationkey"}, {"c_nationkey"});
  auto const join_c = apply_inner_join(join_b, orders, {"c_custkey"}, {"o_custkey"});
  auto const join_d = apply_inner_join(join_c, lineitem, {"o_orderkey"}, {"l_orderkey"});
  auto joined_table =
    apply_inner_join(supplier, join_d, {"s_suppkey", "s_nationkey"}, {"l_suppkey", "n_nationkey"});

  // Calculate and append the `revenue` column
  auto revenue =
    calc_revenue(joined_table->column("l_extendedprice"), joined_table->column("l_discount"));
  (*joined_table).append(revenue, "revenue");

  // Perform the groupby operation
  auto const groupedby_table =
    apply_groupby(joined_table,
                  groupby_context_t{{"n_name"},
                                    {
                                      {"revenue", {{cudf::aggregation::Kind::SUM, "revenue"}}},
                                    }});

  // Perform the order by operation
  auto const orderedby_table =
    apply_orderby(groupedby_table, {"revenue"}, {cudf::order::DESCENDING});

  // Write query result to a parquet file
  orderedby_table->to_parquet("q5.parquet");
  return 0;
}
