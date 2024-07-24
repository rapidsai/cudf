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
#include "utils.hpp"
#include "vocab.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

struct gen_rand_str {
  char* chars;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;

  __host__ __device__ gen_rand_str(char* c) : chars(c), char_dist(32, 137) {}

  __host__ __device__ void operator()(thrust::tuple<cudf::size_type, cudf::size_type> str_begin_end)
  {
    auto begin = thrust::get<0>(str_begin_end);
    auto end   = thrust::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      auto ch = char_dist(engine);
      if (i == end - 1 && ch >= '\x7F') ch = ' ';  // last element ASCII only.
      if (ch >= '\x7F')                            // x7F is at the top edge of ASCII
        chars[i++] = '\xC4';                       // these characters are assigned two bytes
      chars[i] = static_cast<char>(ch + (ch >= '\x7F'));
    }
  }
};

template <typename T>
struct gen_rand_num {
  T lower;
  T upper;

  __host__ __device__ gen_rand_num(T lower, T upper) : lower(lower), upper(upper) {}

  __host__ __device__ T operator()(const int64_t idx) const
  {
    if (cudf::is_integral<T>()) {
      thrust::default_random_engine engine;
      thrust::uniform_int_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    } else {
      thrust::default_random_engine engine;
      thrust::uniform_real_distribution<T> dist(lower, upper);
      engine.discard(idx);
      return dist(engine);
    }
  }
};

std::unique_ptr<cudf::column> gen_rand_str_col(int64_t lower,
                                               int64_t upper,
                                               cudf::size_type num_rows)
{
  rmm::device_uvector<cudf::size_type> offsets(num_rows + 1, cudf::get_default_stream());

  // The first element will always be 0 since it the offset of the first string.
  int64_t initial_offset{0};
  offsets.set_element(0, initial_offset, cudf::get_default_stream());

  // We generate the lengths of the strings randomly for each row and
  // store them from the second element of the offsets vector.
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_rows),
                    offsets.begin() + 1,
                    gen_rand_num<cudf::size_type>(lower, upper));

  // We then calculate the offsets by performing an inclusive scan on this
  // vector.
  thrust::inclusive_scan(
    rmm::exec_policy(cudf::get_default_stream()), offsets.begin(), offsets.end(), offsets.begin());

  // The last element is the total length of all the strings combined using
  // which we allocate the memory for the `chars` vector, that holds the
  // randomly generated characters for the strings.
  auto const total_length = *thrust::device_pointer_cast(offsets.end() - 1);
  rmm::device_uvector<char> chars(total_length, cudf::get_default_stream());

  // We generate the strings in parallel into the `chars` vector using the
  // offsets vector generated above.
  thrust::for_each_n(rmm::exec_policy(cudf::get_default_stream()),
                     thrust::make_zip_iterator(offsets.begin(), offsets.begin() + 1),
                     num_rows,
                     gen_rand_str(chars.data()));

  return cudf::make_strings_column(
    num_rows,
    std::make_unique<cudf::column>(std::move(offsets), rmm::device_buffer{}, 0),
    chars.release(),
    0,
    rmm::device_buffer{});
}

template <typename T>
std::unique_ptr<cudf::column> gen_rand_num_col(T lower, T upper, cudf::size_type count)
{
  cudf::data_type type;
  if (cudf::is_integral<T>()) {
    type = cudf::data_type{cudf::type_id::INT64};
  } else {
    type = cudf::data_type{cudf::type_id::FLOAT64};
  }
  auto col = cudf::make_numeric_column(
    type, count, cudf::mask_state::UNALLOCATED, cudf::get_default_stream());
  thrust::transform(rmm::exec_policy(cudf::get_default_stream()),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(count),
                    col->mutable_view().begin<T>(),
                    gen_rand_num<T>(lower, upper));
  return col;
}

/**
 * @brief Generate a primary key column
 *
 * @param start The starting value of the primary key
 * @param num_rows The number of rows in the column
 */
std::unique_ptr<cudf::column> gen_primary_key_col(int64_t start, int64_t num_rows)
{
  auto const init = cudf::numeric_scalar<int64_t>(start);
  auto const step = cudf::numeric_scalar<int64_t>(1);
  return cudf::sequence(num_rows, init, step);
}

/**
 * @brief Generate a column where all the rows have the same string value
 *
 * @param value The string value
 * @param num_rows The length of the column
 */
std::unique_ptr<cudf::column> gen_repeat_str_col(std::string value, int64_t num_rows)
{
  auto const indices = rmm::device_uvector<cudf::string_view>(num_rows, cudf::get_default_stream());
  auto const empty_str_col =
    cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), cudf::get_default_stream());
  auto const scalar  = cudf::string_scalar(value);
  auto scalar_repeat = cudf::fill(empty_str_col->view(), 0, num_rows, scalar);
  return scalar_repeat;
}

std::unique_ptr<cudf::column> gen_rand_str_col_from_set(std::vector<std::string> string_set,
                                                        int64_t num_rows)
{
  // Build a vocab table of random strings to choose from
  auto const keys = gen_primary_key_col(0, string_set.size());
  auto const values =
    cudf::test::strings_column_wrapper(string_set.begin(), string_set.end()).release();
  auto const vocab_table = cudf::table_view({keys->view(), values->view()});

  // Build a single column table containing `num_rows` random numbers
  auto const rand_keys       = gen_rand_num_col<int64_t>(0, string_set.size() - 1, num_rows);
  auto const rand_keys_table = cudf::table_view({rand_keys->view()});

  auto const joined_table =
    perform_left_join(rand_keys_table, vocab_table, {0}, {0}, cudf::null_equality::EQUAL);
  return std::make_unique<cudf::column>(joined_table->get_column(2));
}

// NOTE: Incomplete table
void generate_lineitem(int64_t scale_factor)
{
  cudf::size_type const num_rows = 1'500'000 * scale_factor;

  // Generate the `l_partkey` column
  auto const l_partkey = gen_rand_num_col<int64_t>(1, 200'000 * scale_factor, num_rows);

  // Generate the `l_quantity` column
  auto const l_quantity = gen_rand_num_col<int64_t>(1, 50, num_rows);

  // Generate the `l_discount` column
  auto const l_discount = gen_rand_num_col<double>(0.0, 0.10, num_rows);

  // Generate the `l_tax` column
  auto const l_tax = gen_rand_num_col<double>(0.0, 0.08, num_rows);

  // Generate the `l_comment` column
  auto const l_comment = gen_rand_str_col(10, 43, num_rows);

  // Generate the `l_shipinstruct` column
  auto const l_shipinstruct = gen_rand_str_col_from_set(vocab_instructions, num_rows);

  // Generate the `l_shipmode` column
  auto const l_shipmode = gen_rand_str_col_from_set(vocab_modes, num_rows);

  auto lineitem = cudf::table_view({l_partkey->view(),
                                    l_quantity->view(),
                                    l_discount->view(),
                                    l_tax->view(),
                                    l_comment->view(),
                                    l_shipinstruct->view(),
                                    l_shipmode->view()});

  write_parquet(lineitem,
                "lineitem.parquet",
                {"l_quantity", "l_discount", "l_tax", "l_comment", "l_shipinstruct", "l_shipmode"});
}

// NOTE: Incomplete table
void generate_orders(int64_t scale_factor)
{
  cudf::size_type const num_rows = 1'500'000 * scale_factor;

  // Generate the `o_orderpriority` column
  auto const o_orderpriority = gen_rand_str_col_from_set(vocab_priorities, num_rows);

  // Generate the `o_shippriority` column
  auto const empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               cudf::get_default_stream());
  auto const o_shippriority =
    cudf::fill(empty->view(), 0, num_rows, cudf::numeric_scalar<int64_t>(0));

  // Generate the `o_comment` column
  auto const o_comment = gen_rand_str_col(19, 78, num_rows);

  auto const orders =
    cudf::table_view({o_orderpriority->view(), o_shippriority->view(), o_comment->view()});

  write_parquet(orders, "orders.parquet", {"o_orderpriority", "o_shippriority", "o_comment"});
}

std::unique_ptr<cudf::column> calc_ps_suppkey(cudf::column_view const& ps_partkey,
                                              int64_t const& scale_factor,
                                              int64_t const& num_rows)
{
  // Generating the `s` col
  auto s_empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           num_rows,
                                           cudf::mask_state::UNALLOCATED,
                                           cudf::get_default_stream());

  auto s =
    cudf::fill(s_empty->view(), 0, num_rows, cudf::numeric_scalar<int64_t>(10000 * scale_factor));

  // Generating the `i` col
  auto seq = gen_primary_key_col(0, num_rows);
  auto i   = cudf::binary_operation(seq->view(),
                                  cudf::numeric_scalar<int64_t>(4),
                                  cudf::binary_operator::MOD,
                                  cudf::data_type{cudf::type_id::INT64});

  // Create a table view out of `p_partkey`, `s`, and `i`
  auto table = cudf::table_view({ps_partkey, s->view(), i->view()});

  // Create the AST expression
  auto scalar_1  = cudf::numeric_scalar<int64_t>(1);
  auto scalar_4  = cudf::numeric_scalar<int64_t>(4);
  auto literal_1 = cudf::ast::literal(scalar_1);
  auto literal_4 = cudf::ast::literal(scalar_4);

  auto ps_partkey_col_ref = cudf::ast::column_reference(0);
  auto s_col_ref          = cudf::ast::column_reference(1);
  auto i_col_ref          = cudf::ast::column_reference(2);

  // (int)(ps_partkey - 1)/s
  auto expr_a = cudf::ast::operation(cudf::ast::ast_operator::SUB, ps_partkey_col_ref, literal_1);
  auto expr_b = cudf::ast::operation(cudf::ast::ast_operator::DIV, expr_a, s_col_ref);
  auto expr_b_casted = cudf::ast::operation(cudf::ast::ast_operator::CAST_TO_INT64, expr_b);

  // s/4
  auto expr_c = cudf::ast::operation(cudf::ast::ast_operator::DIV, s_col_ref, literal_4);

  // (s/4 + (int)(ps_partkey - 1)/s)
  auto expr_d = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_c, expr_b_casted);

  // (i * (s/4 + (int)(ps_partkey - 1)/s))
  auto expr_e = cudf::ast::operation(cudf::ast::ast_operator::MUL, i_col_ref, expr_d);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s)))
  auto expr_f = cudf::ast::operation(cudf::ast::ast_operator::ADD, ps_partkey_col_ref, expr_e);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s
  auto expr_g = cudf::ast::operation(cudf::ast::ast_operator::MOD, expr_f, s_col_ref);

  // (ps_partkey + (i * (s/4 + (int)(ps_partkey - 1)/s))) % s + 1
  auto final_expr = cudf::ast::operation(cudf::ast::ast_operator::ADD, expr_g, literal_1);

  // Execute the AST expression
  auto ps_suppkey = cudf::compute_column(table, final_expr);
  return ps_suppkey;
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
  auto const p_partkey      = gen_primary_key_col(1, num_rows_part);
  auto const rep_freq_empty = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                                        num_rows_part,
                                                        cudf::mask_state::UNALLOCATED,
                                                        cudf::get_default_stream());
  auto const rep_freq =
    cudf::fill(rep_freq_empty->view(), 0, num_rows_part, cudf::numeric_scalar<int64_t>(4));
  auto const rep_table  = cudf::repeat(cudf::table_view({p_partkey->view()}), rep_freq->view());
  auto const ps_partkey = rep_table->get_column(0);

  // Generate the `ps_suppkey` column
  auto const ps_suppkey = calc_ps_suppkey(ps_partkey.view(), scale_factor, num_rows);

  // Generate the `p_availqty` column
  auto const ps_availqty = gen_rand_num_col<int64_t>(1, 9999, num_rows);

  // Generate the `p_supplycost` column
  auto const ps_supplycost = gen_rand_num_col<double>(1.0, 1000.0, num_rows);

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const ps_comment = gen_rand_str_col(49, 198, num_rows);

  auto partsupp = cudf::table_view({ps_partkey.view(),
                                    ps_suppkey->view(),
                                    ps_availqty->view(),
                                    ps_supplycost->view(),
                                    ps_comment->view()});
  write_parquet(partsupp, "partsupp.parquet", schema_partsupp);
}

std::unique_ptr<cudf::column> calc_p_retailprice(cudf::column_view const& p_partkey)
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
void generate_part(int64_t const& scale_factor,
                   rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                   rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource())
{
  cudf::size_type const num_rows = 200'000 * scale_factor;

  // Generate the `p_partkey` column
  auto const p_partkey = gen_primary_key_col(1, num_rows);

  // Generate the `p_name` column
  auto const p_name_a     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_b     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_c     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_d     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_e     = gen_rand_str_col_from_set(vocab_p_name, num_rows);
  auto const p_name_parts = cudf::table_view(
    {p_name_a->view(), p_name_b->view(), p_name_c->view(), p_name_d->view(), p_name_e->view()});
  auto const p_name = cudf::strings::concatenate(p_name_parts, cudf::string_scalar(" "));

  // Generate the `p_mfgr` column
  auto const mfgr_repeat         = gen_repeat_str_col("Manufacturer#", num_rows);
  auto const random_values_m     = gen_rand_num_col<int64_t>(1, 5, num_rows);
  auto const random_values_m_str = cudf::strings::from_integers(random_values_m->view());
  auto const p_mfgr              = cudf::strings::concatenate(
    cudf::table_view({mfgr_repeat->view(), random_values_m_str->view()}));

  // Generate the `p_brand` column
  auto const brand_repeat        = gen_repeat_str_col("Brand#", num_rows);
  auto const random_values_n     = gen_rand_num_col<int64_t>(1, 5, num_rows);
  auto const random_values_n_str = cudf::strings::from_integers(random_values_n->view());
  auto const p_brand             = cudf::strings::concatenate(cudf::table_view(
    {brand_repeat->view(), random_values_m_str->view(), random_values_n_str->view()}));

  // Generate the `p_type` column
  auto const p_type = gen_rand_str_col_from_set(gen_vocab_types(), num_rows);

  // Generate the `p_size` column
  auto const p_size = gen_rand_num_col<int64_t>(1, 50, num_rows);

  // Generate the `p_container` column
  auto const p_container = gen_rand_str_col_from_set(gen_vocab_containers(), num_rows);

  // Generate the `p_retailprice` column
  auto const p_retailprice = calc_p_retailprice(p_partkey->view());

  // Generate the `p_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const p_comment = gen_rand_str_col(5, 22, num_rows);

  // Create the `part` table
  auto part = cudf::table_view({p_partkey->view(),
                                p_name->view(),
                                p_mfgr->view(),
                                p_brand->view(),
                                p_type->view(),
                                p_size->view(),
                                p_container->view(),
                                p_retailprice->view(),
                                p_comment->view()});

  write_parquet(part, "part.parquet", schema_part);
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
  auto const n_nationkey = gen_primary_key_col(0, num_rows);

  // Generate the `n_name` column
  auto const n_name =
    cudf::test::strings_column_wrapper(
      {"ALGERIA",      "ARGENTINA",  "BRAZIL",  "CANADA",         "EGYPT",
       "ETHIOPIA",     "FRANCE",     "GERMANY", "INDIA",          "INDONESIA",
       "IRAN",         "IRAQ",       "JAPAN",   "JORDAN",         "KENYA",
       "MOROCCO",      "MOZAMBIQUE", "PERU",    "CHINA",          "ROMANIA",
       "SAUDI ARABIA", "VIETNAM",    "RUSSIA",  "UNITED KINGDOM", "UNITED STATES"})
      .release();

  // Generate the `n_regionkey` column
  thrust::host_vector<int64_t> const region_keys     = {0, 1, 1, 1, 4, 0, 3, 3, 2, 2, 4, 4, 2,
                                                        4, 0, 0, 0, 1, 2, 3, 4, 2, 3, 3, 1};
  thrust::device_vector<int64_t> const d_region_keys = region_keys;

  auto n_regionkey = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                               num_rows,
                                               cudf::mask_state::UNALLOCATED,
                                               cudf::get_default_stream());
  thrust::copy(rmm::exec_policy(cudf::get_default_stream()),
               d_region_keys.begin(),
               d_region_keys.end(),
               n_regionkey->mutable_view().begin<int64_t>());

  // Generate the `n_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const n_comment = gen_rand_str_col(31, 114, num_rows);

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
  auto const r_regionkey = gen_primary_key_col(0, num_rows);

  // Generate the `r_name` column
  auto const r_name =
    cudf::test::strings_column_wrapper({"AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"})
      .release();

  // Generate the `r_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const r_comment = gen_rand_str_col(31, 115, num_rows);

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
  auto const c_custkey = gen_primary_key_col(1, num_rows);

  // Generate the `c_name` column
  auto const customer_repeat = gen_repeat_str_col("Customer#", num_rows);
  auto const c_custkey_str   = cudf::strings::from_integers(c_custkey->view());
  auto const c_custkey_str_padded =
    cudf::strings::pad(c_custkey_str->view(), 9, cudf::strings::side_type::LEFT, "0");
  auto const c_name = cudf::strings::concatenate(
    cudf::table_view({customer_repeat->view(), c_custkey_str_padded->view()}));

  // Generate the `c_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const c_address = gen_rand_str_col(10, 40, num_rows);

  // Generate the `c_nationkey` column
  auto const c_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows);

  // Generate the `c_phone` column
  auto const c_phone_a =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(10, 34, num_rows)->view());
  auto const c_phone_b =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(100, 999, num_rows)->view());
  auto const c_phone_c =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(100, 999, num_rows)->view());
  auto const c_phone_d =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(1000, 9999, num_rows)->view());
  auto const c_phone_parts =
    cudf::table_view({c_phone_a->view(), c_phone_b->view(), c_phone_c->view(), c_phone_d->view()});
  auto const c_phone = cudf::strings::concatenate(c_phone_parts, cudf::string_scalar("-"));

  // Generate the `c_acctbal` column
  auto const c_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows);

  // Generate the `c_mktsegment` column
  auto const c_mktsegment = gen_rand_str_col_from_set(vocab_segments, num_rows);

  // Generate the `c_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const c_comment = gen_rand_str_col(29, 116, num_rows);

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
  auto const s_suppkey = gen_primary_key_col(1, num_rows);

  // Generate the `s_name` column
  auto const supplier_repeat = gen_repeat_str_col("Supplier#", num_rows);
  auto const s_suppkey_str   = cudf::strings::from_integers(s_suppkey->view());
  auto const s_suppkey_str_padded =
    cudf::strings::pad(s_suppkey_str->view(), 9, cudf::strings::side_type::LEFT, "0");
  auto const s_name = cudf::strings::concatenate(
    cudf::table_view({supplier_repeat->view(), s_suppkey_str_padded->view()}));

  // Generate the `s_address` column
  // NOTE: This column is not compliant with clause 4.2.2.7 of the TPC-H specification
  auto const s_address = gen_rand_str_col(10, 40, num_rows);

  // Generate the `s_nationkey` column
  auto const s_nationkey = gen_rand_num_col<int64_t>(0, 24, num_rows);

  // Generate the `s_phone` column
  auto const s_phone_part_1 =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(10, 34, num_rows)->view());
  auto const s_phone_part_2 =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(100, 999, num_rows)->view());
  auto const s_phone_part_3 =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(100, 999, num_rows)->view());
  auto const s_phone_part_4 =
    cudf::strings::from_integers(gen_rand_num_col<int64_t>(1000, 9999, num_rows)->view());
  auto const s_phone_parts = cudf::table_view({s_phone_part_1->view(),
                                               s_phone_part_2->view(),
                                               s_phone_part_3->view(),
                                               s_phone_part_4->view()});
  auto const s_phone       = cudf::strings::concatenate(s_phone_parts, cudf::string_scalar("-"));

  // Generate the `s_acctbal` column
  auto const s_acctbal = gen_rand_num_col<double>(-999.99, 9999.99, num_rows);

  // Generate the `s_comment` column
  // NOTE: This column is not compliant with clause 4.2.2.10 of the TPC-H specification
  auto const s_comment = gen_rand_str_col(25, 100, num_rows);

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

  // generate_lineitem(scale_factor);
  // generate_orders(scale_factor);
  generate_partsupp(scale_factor);
  generate_part(scale_factor);
  generate_supplier(scale_factor);
  generate_customer(scale_factor);
  generate_nation(scale_factor);
  generate_region(scale_factor);

  return 0;
}
