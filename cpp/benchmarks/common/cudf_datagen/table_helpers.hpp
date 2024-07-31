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

#include <memory>

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
  // (
  //            90000
  //            +
  //            (
  //                  (P_PARTKEY/10)
  //                      modulo
  //                       20001
  //            )
  //            +
  //            100
  //            *
  //            (P_PARTKEY modulo 1000)
  // )
  // /100
  auto val_a = cudf::binary_operation(p_partkey,
                                      cudf::numeric_scalar<cudf::size_type>(10),
                                      cudf::binary_operator::DIV,
                                      cudf::data_type{cudf::type_id::FLOAT64});

  auto val_b = cudf::binary_operation(val_a->view(),
                                      cudf::numeric_scalar<cudf::size_type>(20001),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_c = cudf::binary_operation(p_partkey,
                                      cudf::numeric_scalar<cudf::size_type>(1000),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_d = cudf::binary_operation(val_c->view(),
                                      cudf::numeric_scalar<cudf::size_type>(100),
                                      cudf::binary_operator::MUL,
                                      cudf::data_type{cudf::type_id::INT64});
  // 90000 + val_b + val_d
  auto val_e = cudf::binary_operation(val_b->view(),
                                      cudf::numeric_scalar<cudf::size_type>(90000),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_f = cudf::binary_operation(val_e->view(),
                                      val_d->view(),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT32});

  return cudf::binary_operation(val_f->view(),
                                cudf::numeric_scalar<cudf::size_type>(100),
                                cudf::binary_operator::DIV,
                                cudf::data_type{cudf::type_id::FLOAT64});
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
  return cudf::compute_column(table, final_expr, mr);
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
  auto const sum_agg           = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  auto const l_num_rows_scalar = cudf::reduce(
    o_orderkey_repeat_freqs, *sum_agg, cudf::data_type{cudf::type_id::INT32}, stream, mr);
  return reinterpret_cast<cudf::numeric_scalar<cudf::size_type>*>(l_num_rows_scalar.get())->value();
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
  return cudf::compute_column(table, final_expr, mr);
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
