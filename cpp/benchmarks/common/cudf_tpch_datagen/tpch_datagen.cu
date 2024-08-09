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

#include "rand_utilities.cuh"
#include "table_helpers.hpp"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <cudf_benchmark/tpch_datagen.hpp>

namespace cudf {
namespace datagen {

/**
 * @brief Generate a table out of the independent columns of the `orders` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_orders_independent(cudf::size_type const& scale_factor,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  cudf::size_type const o_num_rows = scale_factor * 1'500'000;

  // Generate a primary key column for the `orders` table
  // required for internal operations, but will not be
  // written into the parquet file
  auto o_pkey = gen_primary_key_col<cudf::size_type>(0, o_num_rows, stream, mr);

  // Generate the `o_orderkey` column
  auto o_orderkey = [&]() {
    auto const o_orderkey_candidates =
      gen_primary_key_col<cudf::size_type>(1, 4 * o_num_rows, stream, mr);
    auto const o_orderkey_unsorted = cudf::sample(cudf::table_view({o_orderkey_candidates->view()}),
                                                  o_num_rows,
                                                  cudf::sample_with_replacement::FALSE,
                                                  0,
                                                  stream,
                                                  mr);
    auto const o_orderkey_view =
      cudf::sort_by_key(o_orderkey_unsorted->view(),
                        cudf::table_view({o_orderkey_unsorted->view().column(0)}),
                        {},
                        {},
                        stream,
                        mr)
        ->get_column(0);
    return std::make_unique<cudf::column>(o_orderkey_view);
  }();

  // Generate the `o_custkey` column
  // NOTE: This column does not comply with the specs which
  // specifies that every value % 3 != 0
  auto o_custkey = gen_rand_num_col<cudf::size_type>(1, o_num_rows, o_num_rows, stream, mr);

  // Generate the `o_orderdate` column
  auto o_orderdate_ts = [&]() {
    auto const o_orderdate_year  = gen_rand_str_col_from_set(years, o_num_rows, stream, mr);
    auto const o_orderdate_month = gen_rand_str_col_from_set(months, o_num_rows, stream, mr);
    auto const o_orderdate_day   = gen_rand_str_col_from_set(days, o_num_rows, stream, mr);
    auto const o_orderdate_str   = cudf::strings::concatenate(
      cudf::table_view(
        {o_orderdate_year->view(), o_orderdate_month->view(), o_orderdate_day->view()}),
      cudf::string_scalar("-"),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);

    return cudf::strings::to_timestamps(o_orderdate_str->view(),
                                        cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                        std::string("%Y-%m-%d"),
                                        stream,
                                        mr);
  }();

  // Generate the `o_orderpriority` column
  auto o_orderpriority = gen_rand_str_col_from_set(vocab_priorities, o_num_rows, stream, mr);

  // Generate the `o_clerk` column
  auto o_clerk = [&]() {
    auto const clerk_repeat = gen_rep_str_col("Clerk#", o_num_rows, stream, mr);
    auto const random_c =
      gen_rand_num_col<cudf::size_type>(1, 1'000 * scale_factor, o_num_rows, stream, mr);
    auto const random_c_str        = cudf::strings::from_integers(random_c->view(), stream, mr);
    auto const random_c_str_padded = cudf::strings::zfill(random_c_str->view(), 9, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view({clerk_repeat->view(), random_c_str_padded->view()}),
      cudf::string_scalar(""),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  // Generate the `o_shippriority` column
  auto o_shippriority = [&]() {
    auto const empty = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::INT8}, o_num_rows, cudf::mask_state::UNALLOCATED, stream);
    return cudf::fill(empty->view(), 0, o_num_rows, cudf::numeric_scalar<int8_t>(0), stream, mr);
  }();

  // Generate the `o_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto o_comment = gen_rand_str_col(19, 78, o_num_rows, stream, mr);

  // Generate the `orders_indenpendent` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(o_pkey));
  columns.push_back(std::move(o_orderkey));
  columns.push_back(std::move(o_custkey));
  columns.push_back(std::move(o_orderdate_ts));
  columns.push_back(std::move(o_orderpriority));
  columns.push_back(std::move(o_clerk));
  columns.push_back(std::move(o_shippriority));
  columns.push_back(std::move(o_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `lineitem` table partially
 *
 * @param orders_independent Table with the independent columns of the `orders` table
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_lineitem_partial(cudf::table_view const& orders_independent,
                                                       cudf::size_type const& scale_factor,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const o_num_rows = orders_independent.num_rows();
  // Generate the `lineitem` table. For each row in the `orders` table,
  // we have a random number (between 1 and 7) of rows in the `lineitem` table

  // For each `o_orderkey`, generate a random number (between 1 and 7),
  // which will be the number of rows in the `lineitem` table that will
  // have the same `l_orderkey`
  auto const o_orderkey_repeat_freqs = gen_rand_num_col<int8_t>(1, 7, o_num_rows, stream, mr);

  // Sum up the `o_orderkey_repeat_freqs` to get the number of rows in the
  // `lineitem` table. This is required to generate the independent columns
  // in the `lineitem` table
  auto const l_num_rows = calc_l_cardinality(o_orderkey_repeat_freqs->view(), stream, mr);

  // We create a column, `l_pkey` which will contain the repeated primary keys,
  // `o_pkey` of the `orders` table as per the frequencies in `o_orderkey_repeat_freqs`
  auto const o_pkey = orders_independent.column(0);
  auto const l_pkey =
    cudf::repeat(cudf::table_view({o_pkey}), o_orderkey_repeat_freqs->view(), stream, mr);

  // To generate the base `lineitem` table, we would need to perform a left join
  // between table(o_pkey, o_orderkey, o_orderdate) and table(l_pkey).
  // The column at index 2 in the `l_base` table will comprise the `l_orderkey` column.
  auto const o_orderkey     = orders_independent.column(1);
  auto const o_orderdate_ts = orders_independent.column(3);

  auto const left_table      = cudf::table_view({l_pkey->view()});
  auto const right_table     = cudf::table_view({o_pkey, o_orderkey, o_orderdate_ts});
  auto const l_base_unsorted = perform_left_join(left_table, right_table, {0}, {0}, stream, mr);
  auto const l_base          = cudf::sort_by_key(l_base_unsorted->view(),
                                        cudf::table_view({l_base_unsorted->get_column(2).view()}),
                                                 {},
                                                 {},
                                        stream,
                                        mr);

  // Generate the `l_orderkey` column
  auto l_orderkey = std::make_unique<cudf::column>(l_base->get_column(2));

  // Generate the `l_partkey` column
  auto l_partkey =
    gen_rand_num_col<cudf::size_type>(1, scale_factor * 200'000, l_num_rows, stream, mr);

  // Generate the `l_suppkey` column
  auto l_suppkey = calc_l_suppkey(l_partkey->view(), scale_factor, l_num_rows, stream, mr);

  // Generate the `l_linenumber` column
  auto l_linenumber = gen_rep_seq_col<int8_t>(7, false, l_num_rows, stream, mr);

  // Generate the `l_quantity` column
  auto l_quantity = gen_rand_num_col<int8_t>(1, 50, l_num_rows, stream, mr);

  // Generate the `l_discount` column
  auto l_discount = gen_rand_num_col<double>(0.0, 0.10, l_num_rows, stream, mr);

  // Generate the `l_tax` column
  auto l_tax = gen_rand_num_col<double>(0.0, 0.08, l_num_rows, stream, mr);

  // Get the `l_orderdate` column from the `l_base` table
  auto const ol_orderdate_ts = l_base->get_column(3);

  // Generate the `l_shipdate` column
  auto l_shipdate_ts = [&]() {
    auto const l_shipdate_rand_add_days = gen_rand_num_col<int8_t>(1, 121, l_num_rows, stream, mr);
    return add_calendrical_days(
      ol_orderdate_ts.view(), l_shipdate_rand_add_days->view(), stream, mr);
  }();

  // Generate the `l_commitdate` column
  auto l_commitdate_ts = [&]() {
    auto const l_commitdate_rand_add_days =
      gen_rand_num_col<int8_t>(30, 90, l_num_rows, stream, mr);
    return add_calendrical_days(
      ol_orderdate_ts.view(), l_commitdate_rand_add_days->view(), stream, mr);
  }();

  // Generate the `l_receiptdate` column
  auto l_receiptdate_ts = [&]() {
    auto const l_receiptdate_rand_add_days =
      gen_rand_num_col<int8_t>(1, 30, l_num_rows, stream, mr);
    return add_calendrical_days(
      l_shipdate_ts->view(), l_receiptdate_rand_add_days->view(), stream, mr);
  }();

  // Define the current date as per clause 4.2.2.12 of the TPC-H specification
  constexpr cudf::size_type current_date_days_since_epoch = 9'298;
  auto current_date =
    cudf::timestamp_scalar<cudf::timestamp_D>(current_date_days_since_epoch, true);
  auto current_date_literal = cudf::ast::literal(current_date);

  // Generate the `l_returnflag` column
  // if `l_receiptdate` <= current_date then "R" or "A" else "N"
  auto l_returnflag = [&]() {
    auto const col_ref = cudf::ast::column_reference(0);
    auto const pred =
      cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, col_ref, current_date_literal);
    auto const binary_mask =
      cudf::compute_column(cudf::table_view({l_receiptdate_ts->view()}), pred, stream, mr);

    auto const multiplier     = gen_rep_seq_col<int8_t>(2, false, l_num_rows, stream, mr);
    auto const ternary_mask   = cudf::binary_operation(binary_mask->view(),
                                                     multiplier->view(),
                                                     cudf::binary_operator::MUL,
                                                     cudf::data_type{cudf::type_id::INT8},
                                                     stream,
                                                     mr);
    auto const indices        = cudf::test::fixed_width_column_wrapper<int8_t>({0, 1, 2}).release();
    auto const keys           = cudf::test::strings_column_wrapper({"N", "A", "R"}).release();
    auto const gather_map     = cudf::table_view({indices->view(), keys->view()});
    auto const gathered_table = cudf::gather(
      gather_map, ternary_mask->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    return std::make_unique<cudf::column>(gathered_table->get_column(1));
  }();

  // Generate the `l_linestatus` column
  // if `l_shipdate` > current_date then "F" else "O"
  auto [l_linestatus, l_linestatus_mask] = [&]() {
    auto const col_ref = cudf::ast::column_reference(0);
    auto const pred =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref, current_date_literal);
    auto mask = cudf::compute_column(cudf::table_view({l_shipdate_ts->view()}), pred, stream, mr);
    auto mask_index_type      = cudf::cast(mask->view(), cudf::data_type{cudf::type_id::INT8});
    auto const indices        = cudf::test::fixed_width_column_wrapper<int8_t>({0, 1}).release();
    auto const keys           = cudf::test::strings_column_wrapper({"O", "F"}).release();
    auto const gather_map     = cudf::table_view({indices->view(), keys->view()});
    auto const gathered_table = cudf::gather(
      gather_map, mask_index_type->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    return std::make_tuple(std::make_unique<cudf::column>(gathered_table->get_column(1)),
                           std::move(mask_index_type));
  }();

  // Generate the `l_shipinstruct` column
  auto l_shipinstruct = gen_rand_str_col_from_set(vocab_instructions, l_num_rows, stream, mr);

  // Generate the `l_shipmode` column
  auto l_shipmode = gen_rand_str_col_from_set(vocab_modes, l_num_rows, stream, mr);

  // Generate the `l_comment` column
  // NOTE: This column is not compliant with
  // clause 4.2.2.10 of the TPC-H specification
  auto l_comment = gen_rand_str_col(10, 43, l_num_rows, stream, mr);

  // Generate the `lineitem_partial` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(l_linestatus_mask));
  columns.push_back(std::move(l_orderkey));
  columns.push_back(std::move(l_partkey));
  columns.push_back(std::move(l_suppkey));
  columns.push_back(std::move(l_linenumber));
  columns.push_back(std::move(l_quantity));
  columns.push_back(std::move(l_discount));
  columns.push_back(std::move(l_tax));
  columns.push_back(std::move(l_shipdate_ts));
  columns.push_back(std::move(l_commitdate_ts));
  columns.push_back(std::move(l_receiptdate_ts));
  columns.push_back(std::move(l_returnflag));
  columns.push_back(std::move(l_linestatus));
  columns.push_back(std::move(l_shipinstruct));
  columns.push_back(std::move(l_shipmode));
  columns.push_back(std::move(l_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> generate_orders_dependent(cudf::table_view const& lineitem,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const l_linestatus_mask = lineitem.column(0);
  auto const l_orderkey        = lineitem.column(1);
  auto const l_discount        = lineitem.column(6);
  auto const l_tax             = lineitem.column(7);
  auto const l_extendedprice   = lineitem.column(16);

  std::vector<std::unique_ptr<cudf::column>> orders_dependent_columns;

  // Generate the `o_totalprice` column
  // We calculate the `charge` column, which is a function of `l_extendedprice`,
  // `l_tax`, and `l_discount` and then group by `l_orderkey` and sum the `charge`
  auto const l_charge = calc_charge(l_extendedprice, l_tax, l_discount, stream, mr);
  auto sum_l_charge   = [&]() {
    auto const keys = cudf::table_view({l_orderkey});
    cudf::groupby::groupby gb(keys);
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.push_back(cudf::groupby::aggregation_request());
    requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests[0].values = l_charge->view();
    auto agg_result    = gb.aggregate(requests);
    return std::move(agg_result.second[0].results[0]);
  }();
  orders_dependent_columns.push_back(std::move(sum_l_charge));

  // Generate the `o_orderstatus` column
  auto o_orderstatus = [&]() {
    auto const keys = cudf::table_view({l_orderkey});
    cudf::groupby::groupby gb(keys);
    std::vector<cudf::groupby::aggregation_request> requests;

    // Perform a `count` aggregation on `l_orderkey`
    requests.push_back(cudf::groupby::aggregation_request());
    requests[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
    requests[0].values = l_orderkey;

    // Perform a `sum` aggregation on `l_linestatus_mask`
    requests.push_back(cudf::groupby::aggregation_request());
    requests[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    requests[1].values = l_linestatus_mask;

    // Perform the aggregations
    auto agg_result = gb.aggregate(requests);

    // Create a `table_view` out of the `l_orderkey`, `count`, and `sum` columns
    auto const count = std::move(agg_result.second[0].results[0]);
    auto const sum   = cudf::cast(
      agg_result.second[1].results[0]->view(), cudf::data_type{cudf::type_id::INT32}, stream, mr);

    auto const table =
      cudf::table_view({agg_result.first->get_column(0).view(), count->view(), sum->view()});

    // Now on this table,
    // if `sum` == `count` then "O",
    // if `sum` == 0, then "F",
    // else "P"

    // So, we first evaluate an expression `sum == count` and generate a boolean mask
    auto const count_ref = cudf::ast::column_reference(1);
    auto const sum_ref   = cudf::ast::column_reference(2);
    auto const expr_a    = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, sum_ref, count_ref);
    auto const mask_a    = cudf::compute_column(table, expr_a);
    auto const o_orderstatus_intermediate =
      cudf::copy_if_else(cudf::string_scalar("O"), cudf::string_scalar("F"), mask_a->view());

    // Then, we evaluate an expression `sum == 0` and generate a boolean mask
    auto zero_scalar        = cudf::numeric_scalar<cudf::size_type>(0);
    auto const zero_literal = cudf::ast::literal(zero_scalar);
    auto const expr_b_left =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, count_ref);
    auto const expr_b_right =
      cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, zero_literal);
    auto const expr_b =
      cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr_b_left, expr_b_right);
    auto const mask_b = cudf::compute_column(table, expr_b);
    return cudf::copy_if_else(
      cudf::string_scalar("P"), o_orderstatus_intermediate->view(), mask_b->view());
  }();
  orders_dependent_columns.push_back(std::move(o_orderstatus));
  return std::make_unique<cudf::table>(std::move(orders_dependent_columns));
}

/**
 * @brief Generate the `partsupp` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_partsupp(cudf::size_type const& scale_factor,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Define the number of rows in the `part` and `partsupp` tables
  cudf::size_type const p_num_rows  = scale_factor * 200'000;
  cudf::size_type const ps_num_rows = scale_factor * 800'000;

  // Generate the `ps_partkey` column
  auto ps_partkey = [&]() {
    auto const p_partkey = gen_primary_key_col<cudf::size_type>(1, p_num_rows, stream, mr);
    auto const rep_table = cudf::repeat(cudf::table_view({p_partkey->view()}), 4, stream, mr);
    return std::make_unique<cudf::column>(rep_table->get_column(0));
  }();

  // Generate the `ps_suppkey` column
  auto ps_suppkey = calc_ps_suppkey(ps_partkey->view(), scale_factor, ps_num_rows, stream, mr);

  // Generate the `ps_availqty` column
  auto ps_availqty = gen_rand_num_col<int16_t>(1, 9999, ps_num_rows, stream, mr);

  // Generate the `ps_supplycost` column
  auto ps_supplycost = gen_rand_num_col<double>(1.0, 1000.0, ps_num_rows, stream, mr);

  // Generate the `ps_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto ps_comment = gen_rand_str_col(49, 198, ps_num_rows, stream, mr);

  // Create the `partsupp` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(ps_partkey));
  columns.push_back(std::move(ps_suppkey));
  columns.push_back(std::move(ps_availqty));
  columns.push_back(std::move(ps_supplycost));
  columns.push_back(std::move(ps_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `part` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_part(cudf::size_type const& scale_factor,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  cudf::size_type const num_rows = scale_factor * 200'000;

  // Generate the `p_partkey` column
  auto p_partkey = gen_primary_key_col<cudf::size_type>(1, num_rows, stream, mr);

  // Generate the `p_name` column
  auto p_name = [&]() {
    auto const p_name_a = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
    auto const p_name_b = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
    auto const p_name_c = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
    auto const p_name_d = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
    auto const p_name_e = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view(
        {p_name_a->view(), p_name_b->view(), p_name_c->view(), p_name_d->view(), p_name_e->view()}),
      cudf::string_scalar(" "),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  // Generate the `p_mfgr` and `p_brand` columns
  auto const random_values_m = gen_rand_num_col<int8_t>(1, 5, num_rows, stream, mr);
  auto const random_values_m_str =
    cudf::strings::from_integers(random_values_m->view(), stream, mr);

  auto const random_values_n = gen_rand_num_col<int8_t>(1, 5, num_rows, stream, mr);
  auto const random_values_n_str =
    cudf::strings::from_integers(random_values_n->view(), stream, mr);

  auto p_mfgr = [&]() {
    auto const mfgr_repeat = gen_rep_str_col("Manufacturer#", num_rows, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view({mfgr_repeat->view(), random_values_m_str->view()}),
      cudf::string_scalar(""),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  auto p_brand = [&]() {
    auto const brand_repeat = gen_rep_str_col("Brand#", num_rows, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view(
        {brand_repeat->view(), random_values_m_str->view(), random_values_n_str->view()}),
      cudf::string_scalar(""),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  // Generate the `p_type` column
  auto p_type = gen_rand_str_col_from_set(gen_vocab_types(), num_rows, stream, mr);

  // Generate the `p_size` column
  auto p_size = gen_rand_num_col<int8_t>(1, 50, num_rows, stream, mr);

  // Generate the `p_container` column
  auto p_container = gen_rand_str_col_from_set(gen_vocab_containers(), num_rows, stream, mr);

  // Generate the `p_retailprice` column
  auto p_retailprice = calc_p_retailprice(p_partkey->view(), stream, mr);

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto p_comment = gen_rand_str_col(5, 22, num_rows, stream, mr);

  // Create the `part` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(p_partkey));
  columns.push_back(std::move(p_name));
  columns.push_back(std::move(p_mfgr));
  columns.push_back(std::move(p_brand));
  columns.push_back(std::move(p_type));
  columns.push_back(std::move(p_size));
  columns.push_back(std::move(p_container));
  columns.push_back(std::move(p_retailprice));
  columns.push_back(std::move(p_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `orders`, `lineitem`, and `part` tables
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::tuple<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>>
generate_orders_lineitem_part(cudf::size_type const& scale_factor,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Generate a table with the independent columns of the `orders` table
  auto orders_independent = generate_orders_independent(scale_factor, stream, mr);

  // Generate the `lineitem` table partially
  auto lineitem_partial =
    generate_lineitem_partial(orders_independent->view(), scale_factor, stream, mr);

  // Generate the `part` table
  auto part = generate_part(scale_factor, stream, mr);

  // Join the `part` and partial `lineitem` tables, then calculate the `l_extendedprice` column,
  // add the column to the `lineitem` table, and write the `lineitem` table to a parquet file

  auto l_extendedprice = [&]() {
    auto const left = cudf::table_view(
      {lineitem_partial->get_column(2).view(), lineitem_partial->get_column(5).view()});
    auto const right  = cudf::table_view({part->get_column(0).view(), part->get_column(7).view()});
    auto joined_table = perform_left_join(left, right, {0}, {0}, stream, mr);
    auto const l_quantity = joined_table->get_column(1);
    auto const l_quantity_fp =
      cudf::cast(l_quantity.view(), cudf::data_type{cudf::type_id::FLOAT64});
    auto const p_retailprice = joined_table->get_column(3);
    return cudf::binary_operation(l_quantity_fp->view(),
                                  p_retailprice.view(),
                                  cudf::binary_operator::MUL,
                                  cudf::data_type{cudf::type_id::FLOAT64},
                                  stream,
                                  mr);
  }();

  auto lineitem_partial_columns = lineitem_partial->release();
  lineitem_partial_columns.push_back(std::move(l_extendedprice));
  auto lineitem_temp = std::make_unique<cudf::table>(std::move(lineitem_partial_columns));

  // Generate the dependent columns of the `orders` table
  // and merge them with the independent columns
  auto orders_dependent = generate_orders_dependent(lineitem_temp->view(), stream, mr);

  auto orders_independent_columns = orders_independent->release();
  orders_independent_columns.erase(orders_independent_columns.begin());
  auto orders_dependent_columns = orders_dependent->release();
  orders_independent_columns.insert(orders_independent_columns.end(),
                                    std::make_move_iterator(orders_dependent_columns.begin()),
                                    std::make_move_iterator(orders_dependent_columns.end()));

  // Create the `orders` table
  auto orders = std::make_unique<cudf::table>(std::move(orders_independent_columns));

  // Create the `lineitem` table
  auto lineitem_temp_columns = lineitem_temp->release();
  lineitem_temp_columns.erase(lineitem_temp_columns.begin());
  auto lineitem = std::make_unique<cudf::table>(std::move(lineitem_temp_columns));

  return std::make_tuple(std::move(orders), std::move(lineitem), std::move(part));
}

/**
 * @brief Generate the `supplier` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_supplier(cudf::size_type const& scale_factor,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Calculate the number of rows based on the scale factor
  cudf::size_type const num_rows = scale_factor * 10'000;

  // Generate the `s_suppkey` column
  auto s_suppkey = gen_primary_key_col<cudf::size_type>(1, num_rows, stream, mr);

  // Generate the `s_name` column
  auto s_name = [&]() {
    auto const supplier_repeat      = gen_rep_str_col("Supplier#", num_rows, stream, mr);
    auto const s_suppkey_str        = cudf::strings::from_integers(s_suppkey->view(), stream, mr);
    auto const s_suppkey_str_padded = cudf::strings::zfill(s_suppkey_str->view(), 9, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view({supplier_repeat->view(), s_suppkey_str_padded->view()}),
      cudf::string_scalar(""),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  // Generate the `s_address` column
  auto s_address = gen_addr_col(num_rows, stream, mr);

  // Generate the `s_nationkey` column
  auto s_nationkey = gen_rand_num_col<int8_t>(0, 24, num_rows, stream, mr);

  // Generate the `s_phone` column
  auto s_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `s_acctbal` column
  auto s_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `s_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto s_comment = gen_rand_str_col(25, 100, num_rows, stream, mr);

  // Create the `supplier` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(s_suppkey));
  columns.push_back(std::move(s_name));
  columns.push_back(std::move(s_address));
  columns.push_back(std::move(s_nationkey));
  columns.push_back(std::move(s_phone));
  columns.push_back(std::move(s_acctbal));
  columns.push_back(std::move(s_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `customer` table
 *
 * @param scale_factor The scale factor to generate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_customer(cudf::size_type const& scale_factor,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Calculate the number of rows based on the scale factor
  cudf::size_type const num_rows = scale_factor * 150'000;

  // Generate the `c_custkey` column
  auto c_custkey = gen_primary_key_col<cudf::size_type>(1, num_rows, stream, mr);

  // Generate the `c_name` column
  auto c_name = [&]() {
    auto const customer_repeat      = gen_rep_str_col("Customer#", num_rows, stream, mr);
    auto const c_custkey_str        = cudf::strings::from_integers(c_custkey->view(), stream, mr);
    auto const c_custkey_str_padded = cudf::strings::zfill(c_custkey_str->view(), 9, stream, mr);
    return cudf::strings::concatenate(
      cudf::table_view({customer_repeat->view(), c_custkey_str_padded->view()}),
      cudf::string_scalar(""),
      cudf::string_scalar("", false),
      cudf::strings::separator_on_nulls::NO,
      stream,
      mr);
  }();

  // Generate the `c_address` column
  auto c_address = gen_addr_col(num_rows, stream, mr);

  // Generate the `c_nationkey` column
  auto c_nationkey = gen_rand_num_col<int8_t>(0, 24, num_rows, stream, mr);

  // Generate the `c_phone` column
  auto c_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `c_acctbal` column
  auto c_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `c_mktsegment` column
  auto c_mktsegment = gen_rand_str_col_from_set(vocab_segments, num_rows, stream, mr);

  // Generate the `c_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto c_comment = gen_rand_str_col(29, 116, num_rows, stream, mr);

  // Create the `customer` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(c_custkey));
  columns.push_back(std::move(c_name));
  columns.push_back(std::move(c_address));
  columns.push_back(std::move(c_nationkey));
  columns.push_back(std::move(c_phone));
  columns.push_back(std::move(c_acctbal));
  columns.push_back(std::move(c_mktsegment));
  columns.push_back(std::move(c_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `nation` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_nation(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Define the number of rows
  constexpr cudf::size_type num_rows = 25;

  // Generate the `n_nationkey` column
  auto n_nationkey = gen_primary_key_col<int8_t>(0, num_rows, stream, mr);

  // Generate the `n_name` column
  auto n_name = cudf::test::strings_column_wrapper(nations.begin(), nations.end()).release();

  // Generate the `n_regionkey` column
  std::vector<int8_t> region_keys{0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                  4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
  auto n_regionkey =
    cudf::test::fixed_width_column_wrapper<int8_t>(region_keys.begin(), region_keys.end())
      .release();

  // Generate the `n_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto n_comment = gen_rand_str_col(31, 114, num_rows, stream, mr);

  // Create the `nation` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(n_nationkey));
  columns.push_back(std::move(n_name));
  columns.push_back(std::move(n_regionkey));
  columns.push_back(std::move(n_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

/**
 * @brief Generate the `region` table
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::table> generate_region(rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << std::endl;

  // Define the number of rows
  constexpr cudf::size_type num_rows = 5;

  // Generate the `r_regionkey` column
  auto r_regionkey = gen_primary_key_col<int8_t>(0, num_rows, stream, mr);

  // Generate the `r_name` column
  auto r_name =
    cudf::test::strings_column_wrapper({"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"})
      .release();

  // Generate the `r_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto r_comment = gen_rand_str_col(31, 115, num_rows, stream, mr);

  // Create the `region` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(r_regionkey));
  columns.push_back(std::move(r_name));
  columns.push_back(std::move(r_comment));
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace datagen
}  // namespace cudf
