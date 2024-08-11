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

#pragma once

#include "rand_utilities.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/filling.hpp>
#include <cudf/join.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

std::vector<std::string> const nations = {
  "ALGERIA", "ARGENTINA", "BRAZIL",         "CANADA",       "EGYPT", "ETHIOPIA", "FRANCE",
  "GERMANY", "INDIA",     "INDONESIA",      "IRAN",         "IRAQ",  "JAPAN",    "JORDAN",
  "KENYA",   "MOROCCO",   "MOZAMBIQUE",     "PERU",         "CHINA", "ROMANIA",  "SAUDI ARABIA",
  "VIETNAM", "RUSSIA",    "UNITED KINGDOM", "UNITED STATES"};

std::vector<std::string> const years  = {"1992", "1993", "1994", "1995", "1996", "1997", "1998"};
std::vector<std::string> const months = {
  "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"};
std::vector<std::string> const days = {
  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10", "11", "12", "13", "14", "15", "16",
  "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"};

std::vector<std::string> const vocab_p_name = {
  "almond",   "antique",   "aquamarine", "azure",      "beige",     "bisque",    "black",
  "blanched", "blue",      "blush",      "brown",      "burlywood", "burnished", "chartreuse",
  "chiffon",  "chocolate", "coral",      "cornflower", "cornsilk",  "cream",     "cyan",
  "dark",     "deep",      "dim",        "dodger",     "drab",      "firebrick", "floral",
  "forest",   "frosted",   "gainsboro",  "ghost",      "goldenrod", "green",     "grey",
  "honeydew", "hot",       "indian",     "ivory",      "khaki",     "lace",      "lavender",
  "lawn",     "lemon",     "light",      "lime",       "linen",     "magenta",   "maroon",
  "medium",   "metallic",  "midnight",   "mint",       "misty",     "moccasin",  "navajo",
  "navy",     "olive",     "orange",     "orchid",     "pale",      "papaya",    "peach",
  "peru",     "pink",      "plum",       "powder",     "puff",      "purple",    "red",
  "rose",     "rosy",      "royal",      "saddle",     "salmon",    "sandy",     "seashell",
  "sienna",   "sky",       "slate",      "smoke",      "snow",      "spring",    "steel",
  "tan",      "thistle",   "tomato",     "turquoise",  "violet",    "wheat",     "white",
  "yellow"};

std::vector<std::string> const vocab_modes = {
  "REG AIR", "AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB"};

std::vector<std::string> const vocab_instructions = {
  "DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"};

std::vector<std::string> const vocab_priorities = {
  "1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};

std::vector<std::string> const vocab_segments = {
  "AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"};

std::vector<std::string> gen_vocab_types()
{
  std::vector<std::string> syllable_a = {
    "STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"};
  std::vector<std::string> syllable_b = {"ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"};
  std::vector<std::string> syllable_c = {"TIN", "NICKEL", "BRASS", "STEEL", "COPPER"};
  std::vector<std::string> syllable_combinations;
  for (auto const& s_a : syllable_a) {
    for (auto const& s_b : syllable_b) {
      for (auto const& s_c : syllable_c) {
        syllable_combinations.push_back(s_a + " " + s_b + " " + s_c);
      }
    }
  }
  return syllable_combinations;
}

std::vector<std::string> gen_vocab_containers()
{
  std::vector<std::string> syllable_a = {"SM", "LG", "MED", "JUMBO", "WRAP"};
  std::vector<std::string> syllable_b = {"CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"};
  std::vector<std::string> syllable_combinations;
  for (auto const& s_a : syllable_a) {
    for (auto const& s_b : syllable_b) {
      syllable_combinations.push_back(s_a + " " + s_b);
    }
  }
  return syllable_combinations;
}

/**
 * @brief Add a column of days to a column of timestamp_days
 *
 * @param timestamp_days The column of timestamp_days
 * @param days The column of days to add
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> add_calendrical_days(cudf::column_view const& timestamp_days,
                                                   cudf::column_view const& days,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const days_duration_type = cudf::cast(days, cudf::data_type{cudf::type_id::DURATION_DAYS});
  auto const data_type          = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  return cudf::binary_operation(
    timestamp_days, days_duration_type->view(), cudf::binary_operator::ADD, data_type, stream, mr);
}

/**
 * @brief Perform a left join operation between two tables
 *
 * @param left_input The left table
 * @param right_input The right table
 * @param left_on The indices of the columns to join on in the left table
 * @param right_on The indices of the columns to join on in the right table
 * @param compare_nulls The null equality comparison
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory
 */
std::unique_ptr<cudf::table> perform_left_join(cudf::table_view const& left_input,
                                               cudf::table_view const& right_input,
                                               std::vector<cudf::size_type> const& left_on,
                                               std::vector<cudf::size_type> const& right_on,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  constexpr auto oob_policy = cudf::out_of_bounds_policy::NULLIFY;
  auto const left_selected  = left_input.select(left_on);
  auto const right_selected = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] =
    cudf::left_join(left_selected, right_selected, cudf::null_equality::EQUAL, mr);

  auto const left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto const right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto const left_indices_col  = cudf::column_view{left_indices_span};
  auto const right_indices_col = cudf::column_view{right_indices_span};

  auto const left_result  = cudf::gather(left_input, left_indices_col, oob_policy, stream, mr);
  auto const right_result = cudf::gather(right_input, right_indices_col, oob_policy, stream, mr);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

/**
 * @brief Generate the `p_retailprice` column of the `part` table
 *
 * @param p_partkey The `p_partkey` column of the `part` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_p_retailprice(cudf::column_view const& p_partkey,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Expression: (90000 + ((p_partkey/10) modulo 20001) + 100 * (p_partkey modulo 1000)) / 100
  auto table             = cudf::table_view({p_partkey});
  auto p_partkey_col_ref = cudf::ast::column_reference(0);

  auto scalar_10    = cudf::numeric_scalar<cudf::size_type>(10);
  auto scalar_100   = cudf::numeric_scalar<cudf::size_type>(100);
  auto scalar_1000  = cudf::numeric_scalar<cudf::size_type>(1000);
  auto scalar_20001 = cudf::numeric_scalar<cudf::size_type>(20001);
  auto scalar_90000 = cudf::numeric_scalar<cudf::size_type>(90000);

  auto literal_10    = cudf::ast::literal(scalar_10);
  auto literal_100   = cudf::ast::literal(scalar_100);
  auto literal_1000  = cudf::ast::literal(scalar_1000);
  auto literal_20001 = cudf::ast::literal(scalar_20001);
  auto literal_90000 = cudf::ast::literal(scalar_90000);

  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::DIV, p_partkey_col_ref, literal_10);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_a, literal_20001);
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::MOD, p_partkey_col_ref, literal_1000);
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::MUL, expr_c, literal_100);
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_b, expr_d);
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_e, literal_90000);
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_f, literal_100);

  // Execute the AST expression
  return cudf::compute_column(table, final_expr, stream, mr);
}

/**
 * @brief Generate the `l_suppkey` column of the `lineitem` table
 *
 * @param l_partkey The `l_partkey` column of the `lineitem` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_l_suppkey(cudf::column_view const& l_partkey,
                                                           cudf::size_type const& scale_factor,
                                                           cudf::size_type const& num_rows,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Expression: (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s))) % s + 1

  // Generate the `s` col
  auto s_empty = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream);

  auto s = cudf::fill(s_empty->view(),
                      0,
                      num_rows,
                      cudf::numeric_scalar<cudf::size_type>(scale_factor * 10'000),
                      stream,
                      mr);

  // Generate the `i` col
  auto i = gen_rep_seq_col<cudf::size_type>(4, true, num_rows, stream, mr);

  // Create a table view out of `l_partkey`, `s`, and `i`
  auto table = cudf::table_view({l_partkey, s->view(), i->view()});

  // Create the AST expression
  auto scalar_1  = cudf::numeric_scalar<cudf::size_type>(1);
  auto scalar_4  = cudf::numeric_scalar<cudf::size_type>(4);
  auto literal_1 = cudf::ast::literal(scalar_1);
  auto literal_4 = cudf::ast::literal(scalar_4);

  auto l_partkey_col_ref = cudf::ast::column_reference(0);
  auto s_col_ref         = cudf::ast::column_reference(1);
  auto i_col_ref         = cudf::ast::column_reference(2);

  // (int)(l_partkey - 1)/s
  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::SUB, l_partkey_col_ref, literal_1);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_a, s_col_ref);

  // s/4
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::DIV, s_col_ref, literal_4);

  // (s/4 + (int)(l_partkey - 1)/s)
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_c, expr_b);

  // (i * (s/4 + (int)(l_partkey - 1)/s))
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::MUL, i_col_ref, expr_d);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s)))
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, l_partkey_col_ref, expr_e);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s))) % s
  auto expr_g = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_f, s_col_ref);

  // (l_partkey + (i * (s/4 + (int)(l_partkey - 1)/s))) % s + 1
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_g, literal_1);

  // Execute the AST expression
  return cudf::compute_column(table, final_expr, stream, mr);
}

/**
 * @brief Generate the `ps_suppkey` column of the `partsupp` table
 *
 * @param ps_partkey The `ps_partkey` column of the `partsupp` table
 * @param scale_factor The scale factor to use
 * @param num_rows The number of rows in the `partsupp` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_ps_suppkey(cudf::column_view const& ps_partkey,
                                                            cudf::size_type const& scale_factor,
                                                            cudf::size_type const& num_rows,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Expression: ps_suppkey = (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s + 1

  // Generate the `s` col
  auto s_empty = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, num_rows, cudf::mask_state::UNALLOCATED, stream);

  auto s = cudf::fill(s_empty->view(),
                      0,
                      num_rows,
                      cudf::numeric_scalar<cudf::size_type>(scale_factor * 10'000),
                      stream,
                      mr);

  // Generate the `i` col
  auto i = gen_rep_seq_col<cudf::size_type>(4, true, num_rows, stream, mr);

  // Create a table view out of `p_partkey`, `s`, and `i`
  auto table = cudf::table_view({ps_partkey, s->view(), i->view()});

  // Create the AST expression
  auto scalar_1  = cudf::numeric_scalar<cudf::size_type>(1);
  auto scalar_4  = cudf::numeric_scalar<cudf::size_type>(4);
  auto literal_1 = cudf::ast::literal(scalar_1);
  auto literal_4 = cudf::ast::literal(scalar_4);

  auto ps_partkey_col_ref = cudf::ast::column_reference(0);
  auto s_col_ref          = cudf::ast::column_reference(1);
  auto i_col_ref          = cudf::ast::column_reference(2);

  // (int)(ps_partkey - 1)/s
  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::SUB, ps_partkey_col_ref, literal_1);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_a, s_col_ref);

  // s/4
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::DIV, s_col_ref, literal_4);

  // (s/4 + (int)(ps_partkey - 1)/s)
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_c, expr_b);

  // (i * (s/4 + (int)(ps_partkey - 1)/s))
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::MUL, i_col_ref, expr_d);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s)))
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, ps_partkey_col_ref, expr_e);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s
  auto expr_g = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_f, s_col_ref);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s + 1
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_g, literal_1);

  // Execute the AST expression
  return cudf::compute_column(table, final_expr, stream, mr);
}

/**
 * @brief Calculate the cardinality of the `lineitem` table
 *
 * @param o_orderkey_repeat_freqs The frequency of each `o_orderkey` value in the `lineitem` table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] cudf::size_type calc_l_cardinality(cudf::column_view const& o_orderkey_repeat_freqs,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const sum_agg           = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const l_num_rows_scalar = cudf::reduce(
    o_orderkey_repeat_freqs, *sum_agg, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  return reinterpret_cast<cudf::numeric_scalar<cudf::size_type>*>(l_num_rows_scalar.get())->value();
}

/**
 * @brief Calculate the charge column for the `lineitem` table
 *
 * @param extendedprice The `l_extendedprice` column
 * @param tax The `l_tax` column
 * @param discount The `l_discount` column
 * @param stream The CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> calc_charge(
  cudf::column_view const& extendedprice,
  cudf::column_view const& tax,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  CUDF_FUNC_RANGE();
  auto const one                = cudf::numeric_scalar<double>(1);
  auto const one_minus_discount = cudf::binary_operation(
    one, discount, cudf::binary_operator::SUB, cudf::data_type{cudf::type_id::FLOAT64}, stream, mr);
  auto disc_price = cudf::binary_operation(extendedprice,
                                           one_minus_discount->view(),
                                           cudf::binary_operator::MUL,
                                           cudf::data_type{cudf::type_id::FLOAT64},
                                           stream,
                                           mr);
  auto const one_plus_tax =
    cudf::binary_operation(one, tax, cudf::binary_operator::ADD, tax.type(), stream, mr);
  return cudf::binary_operation(disc_price->view(),
                                one_plus_tax->view(),
                                cudf::binary_operator::MUL,
                                cudf::data_type{cudf::type_id::FLOAT64},
                                stream,
                                mr);
}

/**
 * @brief Generate a column of random addresses according to TPC-H specification clause 4.2.2.7
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> gen_addr_col(cudf::size_type const& num_rows,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return gen_rand_str_col(10, 40, num_rows, stream, mr);
}

/**
 * @brief Generate a phone number column according to TPC-H specification clause 4.2.2.9
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
[[nodiscard]] std::unique_ptr<cudf::column> gen_phone_col(cudf::size_type const& num_rows,
                                                          rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const part_a =
    cudf::strings::from_integers(gen_rand_num_col<int16_t>(10, 34, num_rows, stream, mr)->view());
  auto const part_b =
    cudf::strings::from_integers(gen_rand_num_col<int16_t>(100, 999, num_rows, stream, mr)->view());
  auto const part_c =
    cudf::strings::from_integers(gen_rand_num_col<int16_t>(100, 999, num_rows, stream, mr)->view());
  auto const part_d = cudf::strings::from_integers(
    gen_rand_num_col<int16_t>(1000, 9999, num_rows, stream, mr)->view());
  auto const phone_parts_table =
    cudf::table_view({part_a->view(), part_b->view(), part_c->view(), part_d->view()});
  return cudf::strings::concatenate(phone_parts_table,
                                    cudf::string_scalar("-"),
                                    cudf::string_scalar("", false),
                                    cudf::strings::separator_on_nulls::NO,
                                    stream,
                                    mr);
}