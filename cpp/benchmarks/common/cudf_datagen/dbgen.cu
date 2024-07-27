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

#include "schema.hpp"
#include "table_helpers.hpp"
#include "vocab.hpp"

void generate_lineitem_and_orders(
  int64_t const& scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const o_num_rows = 1'500'000 * scale_factor;

  // Generate the non-dependent columns of the `orders` table

  // Generate a primary key column for the `orders` table
  // required for internal operations, but will not be
  // written into the parquet file
  auto const o_pkey = gen_primary_key_col(0, o_num_rows, stream, mr);

  // Generate the `o_orderkey` column
  auto const o_orderkey_candidates = gen_primary_key_col(1, 4 * o_num_rows, stream, mr);
  auto const o_orderkey_unsorted   = cudf::sample(cudf::table_view({o_orderkey_candidates->view()}),
                                                o_num_rows,
                                                cudf::sample_with_replacement::FALSE,
                                                0,
                                                stream,
                                                mr);
  auto const o_orderkey =
    cudf::sort_by_key(o_orderkey_unsorted->view(),
                      cudf::table_view({o_orderkey_unsorted->view().column(0)}),
                      {},
                      {},
                      stream,
                      mr)
      ->get_column(0);

  // Generate the `o_custkey` column
  // NOTE: This column does not comply with the specs which
  // specifies that every value % 3 != 0
  auto const o_custkey = gen_rand_num_col<int64_t>(1, o_num_rows, o_num_rows, stream, mr);

  // Generate the `o_orderdate` column
  auto const o_orderdate_year  = gen_rand_str_col_from_set(years, o_num_rows, stream, mr);
  auto const o_orderdate_month = gen_rand_str_col_from_set(months, o_num_rows, stream, mr);
  auto const o_orderdate_day   = gen_rand_str_col_from_set(days, o_num_rows, stream, mr);
  auto const o_orderdate_str   = cudf::strings::concatenate(
    cudf::table_view(
      {o_orderdate_year->view(), o_orderdate_month->view(), o_orderdate_day->view()}),
    cudf::string_scalar("-"),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  auto const o_orderdate_ts =
    cudf::strings::to_timestamps(o_orderdate_str->view(),
                                 cudf::data_type{cudf::type_id::TIMESTAMP_DAYS},
                                 std::string("%Y-%m-%d"),
                                 stream,
                                 mr);

  // Generate the `o_orderpriority` column
  auto const o_orderpriority = gen_rand_str_col_from_set(vocab_priorities, o_num_rows, stream, mr);

  // Generate the `o_clerk` column
  auto const clerk_repeat = gen_rep_str_col("Clerk#", o_num_rows, stream, mr);
  auto const random_c = gen_rand_num_col<int64_t>(1, 1'000 * scale_factor, o_num_rows, stream, mr);
  auto const random_c_str = cudf::strings::from_integers(random_c->view(), stream, mr);
  auto const random_c_str_padded =
    cudf::strings::pad(random_c_str->view(), 9, cudf::strings::side_type::LEFT, "0", stream, mr);
  auto const o_clerk = cudf::strings::concatenate(
    cudf::table_view({clerk_repeat->view(), random_c_str_padded->view()}),
    cudf::string_scalar(""),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Generate the `o_shippriority` column
  auto const empty = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, o_num_rows, cudf::mask_state::UNALLOCATED, stream);
  auto const o_shippriority =
    cudf::fill(empty->view(), 0, o_num_rows, cudf::numeric_scalar<int64_t>(0), stream, mr);

  // Generate the `o_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const o_comment = gen_rand_str_col(19, 78, o_num_rows, stream, mr);

  // Generate the `lineitem` table. For each row in the `orders` table,
  // we have a random number (between 1 and 7) of rows in the `lineitem` table

  // For each `o_orderkey`, generate a random number (between 1 and 7),
  // which will be the number of rows in the `lineitem` table that will
  // have the same `l_orderkey`
  auto const o_orderkey_repeat_freqs = gen_rand_num_col<int64_t>(1, 7, o_num_rows, stream, mr);

  // Sum up the `o_orderkey_repeat_freqs` to get the number of rows in the
  // `lineitem` table. This is required to generate the independent columns
  // in the `lineitem` table
  auto const l_num_rows = calc_l_cardinality(o_orderkey_repeat_freqs->view(), stream, mr);

  // We create a column, `l_pkey` which will contain the repeated primary keys,
  // `_o_pkey` of the `orders` table as per the frequencies in `o_orderkey_repeat_freqs`
  auto const l_pkey =
    cudf::repeat(cudf::table_view({o_pkey->view()}), o_orderkey_repeat_freqs->view(), stream, mr);

  // To generate the base `lineitem` table, we would need to perform a left join
  // between table(o_pkey, o_orderkey, o_orderdate) and table(l_pkey).
  // The column at index 2 in the `l_base` table will comprise the `l_orderkey` column.
  auto const left_table = cudf::table_view({l_pkey->view()});
  auto const right_table =
    cudf::table_view({o_pkey->view(), o_orderkey.view(), o_orderdate_ts->view()});
  auto const l_base_unsorted =
    perform_left_join(left_table, right_table, {0}, {0}, cudf::null_equality::EQUAL);
  auto const l_base = cudf::sort_by_key(l_base_unsorted->view(),
                                        cudf::table_view({l_base_unsorted->get_column(2).view()}),
                                        {},
                                        {},
                                        stream,
                                        mr);

  // Generate the `l_orderkey` column
  auto const l_orderkey = l_base->get_column(2);

  // Generate the `l_partkey` column
  auto const l_partkey =
    gen_rand_num_col<int64_t>(1, 200'000 * scale_factor, l_num_rows, stream, mr);

  // Generate the `l_suppkey` column
  auto const l_suppkey = calc_l_suppkey(l_partkey->view(), scale_factor, l_num_rows, stream, mr);

  // Generate the `l_linenumber` column
  auto const l_linenumber = gen_rep_seq_col(7, l_num_rows, stream, mr);

  // Generate the `l_quantity` column
  auto const l_quantity = gen_rand_num_col<int64_t>(1, 50, l_num_rows, stream, mr);

  // Generate the `l_discount` column
  auto const l_discount = gen_rand_num_col<double>(0.0, 0.10, l_num_rows, stream, mr);

  // Generate the `l_tax` column
  auto const l_tax = gen_rand_num_col<double>(0.0, 0.08, l_num_rows, stream, mr);

  // NOTE: For now, adding months. Need to add a new `add_calendrical_days` function
  // to add days to the `o_orderdate` column. For implementing this column, we use
  // the column at index 3 in the `l_base` table.
  auto const ol_orderdate_ts = l_base->get_column(3);

  // Generate the `l_shipdate` column
  auto const l_shipdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows, stream, mr);
  auto const l_shipdate_ts            = cudf::datetime::add_calendrical_months(
    ol_orderdate_ts.view(), l_shipdate_rand_add_days->view(), mr);

  // Generate the `l_commitdate` column
  auto const l_commitdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows, stream, mr);
  auto const l_commitdate_ts            = cudf::datetime::add_calendrical_months(
    ol_orderdate_ts.view(), l_commitdate_rand_add_days->view(), mr);

  // Generate the `l_receiptdate` column
  auto const l_receiptdate_rand_add_days = gen_rand_num_col<int32_t>(1, 6, l_num_rows, stream, mr);
  auto const l_receiptdate_ts            = cudf::datetime::add_calendrical_months(
    l_shipdate_ts->view(), l_receiptdate_rand_add_days->view(), mr);

  // Define the current date as per clause 4.2.2.12 of the TPC-H specification
  auto current_date =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 6, 17), true);
  auto current_date_literal = cudf::ast::literal(current_date);

  // Generate the `l_returnflag` column
  // if `l_receiptdate` <= current_date then "R" or "A" else "N"
  auto const l_receiptdate_col_ref = cudf::ast::column_reference(0);
  auto const l_returnflag_pred     = cudf::ast::operation(
    cudf::ast::ast_operator::LESS_EQUAL, l_receiptdate_col_ref, current_date_literal);
  auto const l_returnflag_binary_mask =
    cudf::compute_column(cudf::table_view({l_receiptdate_ts->view()}), l_returnflag_pred, mr);
  auto const l_returnflag_binary_mask_int =
    cudf::cast(l_returnflag_binary_mask->view(), cudf::data_type{cudf::type_id::INT64}, stream, mr);

  auto const binarty_to_ternary_multiplier =
    gen_rep_seq_col(2, l_num_rows, stream, mr);  // 1, 2, 1, 2,...
  auto const l_returnflag_ternary_mask =
    cudf::binary_operation(l_returnflag_binary_mask_int->view(),
                           binarty_to_ternary_multiplier->view(),
                           cudf::binary_operator::MUL,
                           cudf::data_type{cudf::type_id::INT64},
                           stream,
                           mr);

  auto const l_returnflag_ternary_mask_str =
    cudf::strings::from_integers(l_returnflag_ternary_mask->view(), stream, mr);

  auto const l_returnflag_replace_target =
    cudf::test::strings_column_wrapper({"0", "1", "2"}).release();
  auto const l_returnflag_replace_with =
    cudf::test::strings_column_wrapper({"N", "A", "R"}).release();

  auto const l_returnflag = cudf::strings::replace(l_returnflag_ternary_mask_str->view(),
                                                   l_returnflag_replace_target->view(),
                                                   l_returnflag_replace_with->view(),
                                                   stream,
                                                   mr);

  // Generate the `l_linestatus` column
  // if `l_shipdate` > current_date then "F" else "O"
  auto const l_shipdate_ts_col_ref = cudf::ast::column_reference(0);
  auto const l_linestatus_pred     = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER, l_shipdate_ts_col_ref, current_date_literal);
  auto const l_linestatus_mask =
    cudf::compute_column(cudf::table_view({l_shipdate_ts->view()}), l_linestatus_pred, mr);

  auto const l_linestatus = cudf::strings::from_booleans(
    l_linestatus_mask->view(), cudf::string_scalar("F"), cudf::string_scalar("O"), stream, mr);

  // Generate the `l_shipinstruct` column
  auto const l_shipinstruct = gen_rand_str_col_from_set(vocab_instructions, l_num_rows, stream, mr);

  // Generate the `l_shipmode` column
  auto const l_shipmode = gen_rand_str_col_from_set(vocab_modes, l_num_rows, stream, mr);

  // Generate the `l_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const l_comment = gen_rand_str_col(10, 43, l_num_rows, stream, mr);

  // Generate the dependent columns of the `orders` table

  // Generate the `o_totalprice` column
  auto const l_charge = calc_charge(l_tax->view(), l_tax->view(), l_discount->view(), stream, mr);
  auto const keys     = cudf::table_view({l_orderkey.view()});
  cudf::groupby::groupby gb(keys);
  std::vector<cudf::groupby::aggregation_request> requests;
  requests.push_back(cudf::groupby::aggregation_request());
  requests[0].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  requests[0].values = l_charge->view();
  auto agg_result    = gb.aggregate(requests);
  auto o_totalprice  = std::move(agg_result.second[0].results[0]);

  // Generate the `o_orderstatus` column
  auto const keys2 = cudf::table_view({l_orderkey.view()});
  cudf::groupby::groupby gb2(keys2);
  std::vector<cudf::groupby::aggregation_request> requests2;
  requests2.push_back(cudf::groupby::aggregation_request());

  requests2[0].aggregations.push_back(cudf::make_count_aggregation<cudf::groupby_aggregation>());
  requests2[0].values = l_orderkey.view();

  requests2.push_back(cudf::groupby::aggregation_request());
  requests2[1].aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  auto const l_linestatus_mask_int =
    cudf::cast(l_linestatus_mask->view(), cudf::data_type{cudf::type_id::INT64});
  requests2[1].values = l_linestatus_mask_int->view();

  auto agg_result2 = gb2.aggregate(requests2);
  auto const count64 =
    cudf::cast(agg_result2.second[0].results[0]->view(), cudf::data_type{cudf::type_id::INT64});
  auto const ttt = cudf::table_view({agg_result2.first->get_column(0).view(),
                                     count64->view(),
                                     agg_result2.second[1].results[0]->view()});

  auto const count_ref = cudf::ast::column_reference(1);
  auto const sum_ref   = cudf::ast::column_reference(2);
  auto expr            = cudf::ast::operation(cudf::ast::ast_operator::EQUAL, sum_ref, count_ref);
  auto const mask      = cudf::compute_column(ttt, expr);

  auto const col_aa =
    cudf::copy_if_else(cudf::string_scalar("O"), cudf::string_scalar("F"), mask->view());

  auto const ttta = cudf::table_view({agg_result2.first->get_column(0).view(),
                                      count64->view(),
                                      agg_result2.second[1].results[0]->view(),
                                      col_aa->view()});

  auto zero_scalar  = cudf::numeric_scalar<int64_t>(0);
  auto zero_literal = cudf::ast::literal(zero_scalar);
  auto expr2_a      = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, count_ref);
  auto expr2_b = cudf::ast::operation(cudf::ast::ast_operator::NOT_EQUAL, sum_ref, zero_literal);
  auto expr2   = cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, expr2_a, expr2_b);

  auto const mask2  = cudf::compute_column(ttt, expr2);
  auto const col_bb = cudf::copy_if_else(cudf::string_scalar("P"), col_aa->view(), mask2->view());
  auto const tttaa  = cudf::table_view({agg_result2.first->get_column(0).view(),
                                        count64->view(),
                                        agg_result2.second[1].results[0]->view(),
                                        col_aa->view(),
                                        col_bb->view()});
  // Write the `orders` table to a parquet file
  auto orders = cudf::table_view({o_orderkey.view(),
                                  o_custkey->view(),
                                  o_totalprice->view(),
                                  o_orderdate_ts->view(),
                                  o_orderpriority->view(),
                                  o_clerk->view(),
                                  o_shippriority->view(),
                                  o_comment->view()});

  write_parquet(orders, "orders.parquet", schema_orders);

  // Write the `lineitem` table to a parquet file
  auto lineitem = cudf::table_view({l_orderkey.view(),
                                    l_partkey->view(),
                                    l_suppkey->view(),
                                    l_linenumber->view(),
                                    l_quantity->view(),
                                    l_discount->view(),
                                    l_tax->view(),
                                    l_shipdate_ts->view(),
                                    l_commitdate_ts->view(),
                                    l_receiptdate_ts->view(),
                                    l_returnflag->view(),
                                    l_linestatus->view(),
                                    l_shipinstruct->view(),
                                    l_shipmode->view(),
                                    l_comment->view()});

  write_parquet(lineitem, "lineitem.parquet", schema_lineitem);
}

/**
 * @brief Generate the `partsupp` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_partsupp(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows_part = 200'000 * scale_factor;
  cudf::size_type const num_rows      = 800'000 * scale_factor;

  // Generate the `ps_partkey` column
  auto const p_partkey      = gen_primary_key_col(1, num_rows_part, stream, mr);
  auto const rep_freq_empty = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows_part, cudf::mask_state::UNALLOCATED, stream);
  auto const rep_freq = cudf::fill(
    rep_freq_empty->view(), 0, num_rows_part, cudf::numeric_scalar<int64_t>(4), stream, mr);
  auto const rep_table =
    cudf::repeat(cudf::table_view({p_partkey->view()}), rep_freq->view(), stream, mr);
  auto const ps_partkey = rep_table->get_column(0);

  // Generate the `ps_suppkey` column
  auto const ps_suppkey = calc_ps_suppkey(ps_partkey.view(), scale_factor, num_rows, stream, mr);

  // Generate the `p_availqty` column
  auto const ps_availqty = gen_rand_num_col<int64_t>(1, 9999, num_rows, stream, mr);

  // Generate the `p_supplycost` column
  auto const ps_supplycost = gen_rand_num_col<double>(1.0, 1000.0, num_rows, stream, mr);

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const ps_comment = gen_rand_str_col(49, 198, num_rows, stream, mr);

  auto partsupp = cudf::table_view({ps_partkey.view(),
                                    ps_suppkey->view(),
                                    ps_availqty->view(),
                                    ps_supplycost->view(),
                                    ps_comment->view()});
  write_parquet(partsupp, "partsupp.parquet", schema_partsupp);
}

std::unique_ptr<cudf::column> calc_p_retailprice(cudf::column_view const& p_partkey,
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
                                      cudf::numeric_scalar<int64_t>(10),
                                      cudf::binary_operator::DIV,
                                      cudf::data_type{cudf::type_id::FLOAT64});

  auto val_b = cudf::binary_operation(val_a->view(),
                                      cudf::numeric_scalar<int64_t>(20001),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_c = cudf::binary_operation(p_partkey,
                                      cudf::numeric_scalar<int64_t>(1000),
                                      cudf::binary_operator::MOD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_d = cudf::binary_operation(val_c->view(),
                                      cudf::numeric_scalar<int64_t>(100),
                                      cudf::binary_operator::MUL,
                                      cudf::data_type{cudf::type_id::INT64});
  // 90000 + val_b + val_d
  auto val_e = cudf::binary_operation(val_b->view(),
                                      cudf::numeric_scalar<int64_t>(90000),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto val_f = cudf::binary_operation(val_e->view(),
                                      val_d->view(),
                                      cudf::binary_operator::ADD,
                                      cudf::data_type{cudf::type_id::INT64});

  auto p_retailprice = cudf::binary_operation(val_f->view(),
                                              cudf::numeric_scalar<int64_t>(100),
                                              cudf::binary_operator::DIV,
                                              cudf::data_type{cudf::type_id::FLOAT64});

  return p_retailprice;
}

/**
 * @brief Generate the `part` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
std::unique_ptr<cudf::table> generate_part(
  int64_t const& scale_factor,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 200'000 * scale_factor;

  // Generate the `p_partkey` column
  auto const p_partkey = gen_primary_key_col(1, num_rows, stream, mr);

  // Generate the `p_name` column
  auto const p_name_a = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
  auto const p_name_b = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
  auto const p_name_c = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
  auto const p_name_d = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
  auto const p_name_e = gen_rand_str_col_from_set(vocab_p_name, num_rows, stream, mr);
  auto const p_name   = cudf::strings::concatenate(
    cudf::table_view(
      {p_name_a->view(), p_name_b->view(), p_name_c->view(), p_name_d->view(), p_name_e->view()}),
    cudf::string_scalar(" "),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Generate the `p_mfgr` column
  auto const mfgr_repeat     = gen_rep_str_col("Manufacturer#", num_rows, stream, mr);
  auto const random_values_m = gen_rand_num_col<int64_t>(1, 5, num_rows, stream, mr);
  auto const random_values_m_str =
    cudf::strings::from_integers(random_values_m->view(), stream, mr);
  auto const p_mfgr =
    cudf::strings::concatenate(cudf::table_view({mfgr_repeat->view(), random_values_m_str->view()}),
                               cudf::string_scalar(""),
                               cudf::string_scalar("", false),
                               cudf::strings::separator_on_nulls::YES,
                               stream,
                               mr);

  // Generate the `p_brand` column
  auto const brand_repeat    = gen_rep_str_col("Brand#", num_rows, stream, mr);
  auto const random_values_n = gen_rand_num_col<int64_t>(1, 5, num_rows, stream, mr);
  auto const random_values_n_str =
    cudf::strings::from_integers(random_values_n->view(), stream, mr);
  auto const p_brand = cudf::strings::concatenate(
    cudf::table_view(
      {brand_repeat->view(), random_values_m_str->view(), random_values_n_str->view()}),
    cudf::string_scalar(""),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Generate the `p_type` column
  auto const p_type = gen_rand_str_col_from_set(gen_vocab_types(), num_rows, stream, mr);

  // Generate the `p_size` column
  auto const p_size = gen_rand_num_col<int64_t>(1, 50, num_rows, stream, mr);

  // Generate the `p_container` column
  auto const p_container = gen_rand_str_col_from_set(gen_vocab_containers(), num_rows, stream, mr);

  // Generate the `p_retailprice` column
  auto const p_retailprice = calc_p_retailprice(p_partkey->view(), stream, mr);

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const p_comment = gen_rand_str_col(5, 22, num_rows, stream, mr);

  // Create the `part` table
  auto part_view = cudf::table_view({p_partkey->view(),
                                     p_name->view(),
                                     p_mfgr->view(),
                                     p_brand->view(),
                                     p_type->view(),
                                     p_size->view(),
                                     p_container->view(),
                                     p_retailprice->view(),
                                     p_comment->view()});

  return std::make_unique<cudf::table>(part_view);
}

/**
 * @brief Generate the `nation` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_nation(int64_t const& scale_factor,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 25;

  // Generate the `n_nationkey` column
  auto const n_nationkey = gen_primary_key_col(0, num_rows, stream, mr);

  // Generate the `n_name` column
  auto const n_name = cudf::test::strings_column_wrapper(nations.begin(), nations.end()).release();

  // Generate the `n_regionkey` column
  thrust::host_vector<int64_t> const region_keys     = {0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                                        4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
  thrust::device_vector<int64_t> const d_region_keys = region_keys;

  auto n_regionkey = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, num_rows, cudf::mask_state::UNALLOCATED, stream);
  thrust::copy(rmm::exec_policy(stream),
               d_region_keys.begin(),
               d_region_keys.end(),
               n_regionkey->mutable_view().begin<int64_t>());

  // Generate the `n_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const n_comment = gen_rand_str_col(31, 114, num_rows, stream, mr);

  // Create the `nation` table
  auto nation =
    cudf::table_view({n_nationkey->view(), n_name->view(), n_regionkey->view(), n_comment->view()});
  write_parquet(nation, "nation.parquet", schema_nation);
}

/**
 * @brief Generate the `region` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_region(int64_t const& scale_factor,
                     rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                     rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 5;

  // Generate the `r_regionkey` column
  auto const r_regionkey = gen_primary_key_col(0, num_rows, stream, mr);

  // Generate the `r_name` column
  auto const r_name =
    cudf::test::strings_column_wrapper({"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"})
      .release();

  // Generate the `r_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const r_comment = gen_rand_str_col(31, 115, num_rows, stream, mr);

  // Create the `region` table
  auto region = cudf::table_view({r_regionkey->view(), r_name->view(), r_comment->view()});
  write_parquet(region, "region.parquet", schema_region);
}

/**
 * @brief Generate the `customer` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_customer(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 150'000 * scale_factor;

  // Generate the `c_custkey` column
  auto const c_custkey = gen_primary_key_col(1, num_rows, stream, mr);

  // Generate the `c_name` column
  auto const customer_repeat = gen_rep_str_col("Customer#", num_rows, stream, mr);
  auto const c_custkey_str   = cudf::strings::from_integers(c_custkey->view(), stream, mr);
  auto const c_custkey_str_padded =
    cudf::strings::pad(c_custkey_str->view(), 9, cudf::strings::side_type::LEFT, "0", stream, mr);
  auto const c_name = cudf::strings::concatenate(
    cudf::table_view({customer_repeat->view(), c_custkey_str_padded->view()}),
    cudf::string_scalar(""),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Generate the `c_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const c_address = gen_rand_str_col(10, 40, num_rows, stream, mr);

  // Generate the `c_nationkey` column
  auto const c_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows, stream, mr);

  // Generate the `c_phone` column
  auto const c_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `c_acctbal` column
  auto const c_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `c_mktsegment` column
  auto const c_mktsegment = gen_rand_str_col_from_set(vocab_segments, num_rows, stream, mr);

  // Generate the `c_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const c_comment = gen_rand_str_col(29, 116, num_rows, stream, mr);

  // Create the `customer` table
  auto customer = cudf::table_view({c_custkey->view(),
                                    c_name->view(),
                                    c_address->view(),
                                    c_nationkey->view(),
                                    c_phone->view(),
                                    c_acctbal->view(),
                                    c_mktsegment->view(),
                                    c_comment->view()});
  write_parquet(customer, "customer.parquet", schema_customer);
}

/**
 * @brief Generate the `supplier` table
 *
 * @param scale_factor The scale factor to use
 * @param stream The CUDA stream to use
 * @param mr The memory resource to use
 */
void generate_supplier(int64_t const& scale_factor,
                       rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                       rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 10'000 * scale_factor;

  // Generate the `s_suppkey` column
  auto const s_suppkey = gen_primary_key_col(1, num_rows, stream, mr);

  // Generate the `s_name` column
  auto const supplier_repeat = gen_rep_str_col("Supplier#", num_rows, stream, mr);
  auto const s_suppkey_str   = cudf::strings::from_integers(s_suppkey->view(), stream, mr);
  auto const s_suppkey_str_padded =
    cudf::strings::pad(s_suppkey_str->view(), 9, cudf::strings::side_type::LEFT, "0", stream, mr);
  auto const s_name = cudf::strings::concatenate(
    cudf::table_view({supplier_repeat->view(), s_suppkey_str_padded->view()}),
    cudf::string_scalar(""),
    cudf::string_scalar("", false),
    cudf::strings::separator_on_nulls::YES,
    stream,
    mr);

  // Generate the `s_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const s_address = gen_rand_str_col(10, 40, num_rows, stream, mr);

  // Generate the `s_nationkey` column
  auto const s_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows, stream, mr);

  // Generate the `s_phone` column
  auto const s_phone = gen_phone_col(num_rows, stream, mr);

  // Generate the `s_acctbal` column
  auto const s_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows, stream, mr);

  // Generate the `s_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const s_comment = gen_rand_str_col(25, 100, num_rows, stream, mr);

  // Create the `supplier` table
  auto supplier = cudf::table_view({s_suppkey->view(),
                                    s_name->view(),
                                    s_address->view(),
                                    s_nationkey->view(),
                                    s_phone->view(),
                                    s_acctbal->view(),
                                    s_comment->view()});
  write_parquet(supplier, "supplier.parquet", schema_supplier);
}

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::set_current_device_resource(&cuda_mr);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scale_factor>" << std::endl;
    return 1;
  }

  int32_t scale_factor = std::atoi(argv[1]);
  std::cout << "Requested scale factor: " << scale_factor << std::endl;

  generate_lineitem_and_orders(scale_factor);
  generate_partsupp(scale_factor);
  generate_supplier(scale_factor);
  generate_customer(scale_factor);
  generate_nation(scale_factor);
  generate_region(scale_factor);

  return 0;
}
