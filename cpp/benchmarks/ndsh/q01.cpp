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

#include "utilities.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q01.cpp
 * @brief Implement query 1 of the NDS-H benchmark.
 *
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 *
 * select
 *    l_returnflag,
 *    l_linestatus,
 *    sum(l_quantity) as sum_qty,
 *    sum(l_extendedprice) as sum_base_price,
 *    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
 *    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
 *    avg(l_quantity) as avg_qty,
 *    avg(l_extendedprice) as avg_price,
 *    avg(l_discount) as avg_disc,
 *    count(*) as count_order
 * from
 *    lineitem
 * where
 *    l_shipdate <= date '1998-09-02'
 * group by
 *    l_returnflag,
 *    l_linestatus
 * order by
 *    l_returnflag,
 *    l_linestatus;
 */

/**
 * @brief Calculate the discount price column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_disc_price(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  auto const disc_price_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto disc_price            = cudf::binary_operation(extendedprice,
                                           one_minus_discount->view(),
                                           cudf::binary_operator::MUL,
                                           disc_price_type,
                                           stream,
                                           mr);
  return disc_price;
}

/**
 * @brief Calculate the charge column
 *
 * @param tax The tax column
 * @param disc_price The discount price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_charge(
  cudf::column_view const& tax,
  cudf::column_view const& disc_price,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = cudf::numeric_scalar<double>(1);
  auto const one_plus_tax =
    cudf::binary_operation(one, tax, cudf::binary_operator::ADD, tax.type(), stream, mr);
  auto const charge_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto charge            = cudf::binary_operation(
    disc_price, one_plus_tax->view(), cudf::binary_operator::MUL, charge_type, stream, mr);
  return charge;
}

void run_ndsh_q1(nvbench::state& state,
                 std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  // Define the column projections and filter predicate for `lineitem` table
  std::vector<std::string> const lineitem_cols = {"l_returnflag",
                                                  "l_linestatus",
                                                  "l_quantity",
                                                  "l_extendedprice",
                                                  "l_discount",
                                                  "l_shipdate",
                                                  "l_orderkey",
                                                  "l_tax"};
  auto const shipdate_ref                      = cudf::ast::column_reference(std::distance(
    lineitem_cols.begin(), std::find(lineitem_cols.begin(), lineitem_cols.end(), "l_shipdate")));
  auto shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1998, 9, 2), true);
  auto const shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
  auto const lineitem_pred          = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL, shipdate_ref, shipdate_upper_literal);

  // Read out the `lineitem` table from parquet file
  auto lineitem = read_parquet(
    sources.at("lineitem").make_source_info(), lineitem_cols, std::move(lineitem_pred));

  // Calculate the discount price and charge columns and append to lineitem table
  auto disc_price =
    calculate_disc_price(lineitem->column("l_discount"), lineitem->column("l_extendedprice"));
  auto charge = calculate_charge(lineitem->column("l_tax"), disc_price->view());
  (*lineitem).append(disc_price, "disc_price").append(charge, "charge");

  // Perform the group by operation
  auto const groupedby_table = apply_groupby(
    lineitem,
    groupby_context_t{
      {"l_returnflag", "l_linestatus"},
      {
        {"l_extendedprice",
         {{cudf::aggregation::Kind::SUM, "sum_base_price"},
          {cudf::aggregation::Kind::MEAN, "avg_price"}}},
        {"l_quantity",
         {{cudf::aggregation::Kind::SUM, "sum_qty"}, {cudf::aggregation::Kind::MEAN, "avg_qty"}}},
        {"l_discount",
         {
           {cudf::aggregation::Kind::MEAN, "avg_disc"},
         }},
        {"disc_price",
         {
           {cudf::aggregation::Kind::SUM, "sum_disc_price"},
         }},
        {"charge",
         {{cudf::aggregation::Kind::SUM, "sum_charge"},
          {cudf::aggregation::Kind::COUNT_ALL, "count_order"}}},
      }});

  // Perform the order by operation
  auto const orderedby_table = apply_orderby(groupedby_table,
                                             {"l_returnflag", "l_linestatus"},
                                             {cudf::order::ASCENDING, cudf::order::ASCENDING});

  // Write query result to a parquet file
  orderedby_table->to_parquet("q1.parquet");
}

void ndsh_q1(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q1(state, sources); });
}

NVBENCH_BENCH(ndsh_q1).set_name("ndsh_q1").add_float64_axis("scale_factor", {0.01, 0.1, 1});
