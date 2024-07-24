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

#include "../utilities/timer.hpp"
#include "utils.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>

int main(int argc, char const** argv)
{
  auto const args = parse_args(argc, argv);

  // Use a memory pool
  auto resource = create_memory_resource(args.memory_resource_type);
  rmm::mr::set_current_device_resource(resource.get());

  cudf::examples::timer timer;

  // Define the column projection and filter predicate for the `orders` table
  std::vector<std::string> const orders_cols = {"o_orderdate"};
  auto const o_orderdate_ref                 = cudf::ast::column_reference(std::distance(
    orders_cols.begin(), std::find(orders_cols.begin(), orders_cols.end(), "o_orderdate")));
  auto o_orderdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1993, 1, 1), true);
  auto const o_orderdate_lower_limit = cudf::ast::literal(o_orderdate_lower);
  auto const o_orderdate_pred_lower  = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, o_orderdate_ref, o_orderdate_lower_limit);
  auto o_orderdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const o_orderdate_upper_limit = cudf::ast::literal(o_orderdate_upper);
  auto const o_orderdate_pred_upper =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, o_orderdate_ref, o_orderdate_upper_limit);
  auto const orders_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, o_orderdate_pred_lower, o_orderdate_pred_upper);

  // Read out tables from the parquet files by pushing down column projections and filter
  // predicates
  auto const customer =
    read_parquet(args.dataset_dir + "/customer.parquet",
                 {"c_custkey", "c_name", "c_acctbal", "c_address", "c_phone", "c_comment"});
  auto const orders =
    read_parquet(args.dataset_dir + "/orders.parquet", orders_cols, std::move(orders_pred));
  auto const lineitem =
    read_parquet(args.dataset_dir + "/lineitem.parquet",
                 {"l_extendedprice", "l_discount", "l_orderkey", "l_returnflag"});
  auto const nation = read_parquet(args.dataset_dir + "/nation.parquet", {"n_name"});

  // Filter the `lineitem` table using `l_returnflag = `R`'
  auto const l_returnflag_ref = cudf::ast::column_reference(3);
  auto r_scalar               = cudf::string_scalar("R");
  auto const r_literal        = cudf::ast::literal(r_scalar);
  auto const l_returnflag_pred =
    cudf::ast::operation(cudf::ast::ast_operator::EQUAL, l_returnflag_ref, r_literal);

  // Perform the joins
  auto const join_a       = apply_inner_join(nation, customer, {"n_nationkey"}, {"c_nationkey"});
  auto const join_b       = apply_inner_join(join_a, orders, {"c_custkey"}, {"o_custkey"});
  auto const joined_table = apply_inner_join(join_b, lineitem, {"o_orderkey"}, {"l_orderkey"});
}
