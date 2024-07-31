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
#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/datetime.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/join.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/lists/filling.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/convert/convert_booleans.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/unary.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
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

#include <ctime>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

void write_parquet(std::unique_ptr<cudf::table> tbl,
                   std::string const& path,
                   std::vector<std::string> const& col_names)
{
  CUDF_FUNC_RANGE();
  std::cout << __func__ << " " << path << std::endl;
  auto const sink_info = cudf::io::sink_info(path);
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto& col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info            = col_name_infos;
  auto const table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::parquet_writer_options::builder(sink_info, tbl->view());
  builder.metadata(table_input_metadata);
  auto const options = builder.build();
  cudf::io::write_parquet(options);
}

std::unique_ptr<cudf::table> perform_inner_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  CUDF_FUNC_RANGE();
  constexpr auto oob_policy                          = cudf::out_of_bounds_policy::DONT_CHECK;
  auto const left_selected                           = left_input.select(left_on);
  auto const right_selected                          = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] = cudf::inner_join(
    left_selected, right_selected, compare_nulls, rmm::mr::get_current_device_resource());

  auto const left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto const right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto const left_indices_col  = cudf::column_view{left_indices_span};
  auto const right_indices_col = cudf::column_view{right_indices_span};

  auto const left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto const right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

std::unique_ptr<cudf::table> perform_left_join(
  cudf::table_view const& left_input,
  cudf::table_view const& right_input,
  std::vector<cudf::size_type> const& left_on,
  std::vector<cudf::size_type> const& right_on,
  cudf::null_equality compare_nulls = cudf::null_equality::EQUAL)
{
  CUDF_FUNC_RANGE();
  constexpr auto oob_policy                          = cudf::out_of_bounds_policy::NULLIFY;
  auto const left_selected                           = left_input.select(left_on);
  auto const right_selected                          = right_input.select(right_on);
  auto const [left_join_indices, right_join_indices] = cudf::left_join(
    left_selected, right_selected, compare_nulls, rmm::mr::get_current_device_resource());

  auto const left_indices_span  = cudf::device_span<cudf::size_type const>{*left_join_indices};
  auto const right_indices_span = cudf::device_span<cudf::size_type const>{*right_join_indices};

  auto const left_indices_col  = cudf::column_view{left_indices_span};
  auto const right_indices_col = cudf::column_view{right_indices_span};

  auto const left_result  = cudf::gather(left_input, left_indices_col, oob_policy);
  auto const right_result = cudf::gather(right_input, right_indices_col, oob_policy);

  auto joined_cols = left_result->release();
  auto right_cols  = right_result->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

constexpr size_t POOL_SIZE = 2147483648;

// RMM memory resource creation utilities
inline auto make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }
inline auto make_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_cuda(), POOL_SIZE);
}
inline auto make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }
inline auto make_managed_pool()
{
  return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(make_managed(), POOL_SIZE);
}
inline std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(
  std::string const& mode)
{
  if (mode == "cuda") return make_cuda();
  if (mode == "pool") return make_pool();
  if (mode == "managed") return make_managed();
  if (mode == "managed_pool") return make_managed_pool();
  CUDF_FAIL("Unknown rmm_mode parameter: " + mode +
            "\nExpecting: cuda, pool, managed, or managed_pool");
}

// Convert a C++ type to a cudf::data_type
template <typename T>
cudf::data_type get_cudf_type()
{
  if (std::is_same<T, bool>::value) return cudf::data_type{cudf::type_id::BOOL8};
  if (std::is_same<T, int8_t>::value) return cudf::data_type{cudf::type_id::INT8};
  if (std::is_same<T, int16_t>::value) return cudf::data_type{cudf::type_id::INT16};
  if (std::is_same<T, int32_t>::value) return cudf::data_type{cudf::type_id::INT32};
  if (std::is_same<T, int64_t>::value) return cudf::data_type{cudf::type_id::INT64};
  if (std::is_same<T, float>::value) return cudf::data_type{cudf::type_id::FLOAT32};
  if (std::is_same<T, double>::value) return cudf::data_type{cudf::type_id::FLOAT64};
  CUDF_FAIL("Unsupported type for cudf::data_type");
}

/**
 * @brief Generate the `std::tm` structure from year, month, and day
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
std::tm make_tm(int year, int month, int day)
{
  std::tm tm{};
  tm.tm_year = year - 1900;
  tm.tm_mon  = month - 1;
  tm.tm_mday = day;
  return tm;
}

/**
 * @brief Calculate the number of days since the UNIX epoch
 *
 * @param year The year
 * @param month The month
 * @param day The day
 */
int32_t days_since_epoch(int year, int month, int day)
{
  std::tm tm             = make_tm(year, month, day);
  std::tm epoch          = make_tm(1970, 1, 1);
  std::time_t time       = std::mktime(&tm);
  std::time_t epoch_time = std::mktime(&epoch);
  double diff            = std::difftime(time, epoch_time) / (60 * 60 * 24);
  return static_cast<int32_t>(diff);
}

// Functor for generating random strings
struct gen_rand_str {
  char* chars;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;

  __host__ __device__ gen_rand_str(char* c) : chars(c), char_dist(44, 122) {}

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

// Functor for generating random numbers
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

/**
 * @brief Generate a column of random strings
 *
 * @param lower The lower bound of the length of the strings
 * @param upper The upper bound of the length of the strings
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rand_str_col(cudf::size_type const& lower,
                                               cudf::size_type const& upper,
                                               cudf::size_type const& num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  rmm::device_uvector<cudf::size_type> offsets(num_rows + 1, stream);

  // The first element will always be 0 since it the offset of the first string.
  cudf::size_type initial_offset{0};
  offsets.set_element(0, initial_offset, stream);

  // We generate the lengths of the strings randomly for each row and
  // store them from the second element of the offsets vector.
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(num_rows),
                    offsets.begin() + 1,
                    gen_rand_num<cudf::size_type>(lower, upper));

  // We then calculate the offsets by performing an inclusive scan on this
  // vector.
  thrust::inclusive_scan(rmm::exec_policy(stream), offsets.begin(), offsets.end(), offsets.begin());

  // The last element is the total length of all the strings combined using
  // which we allocate the memory for the `chars` vector, that holds the
  // randomly generated characters for the strings.
  auto const total_length = *thrust::device_pointer_cast(offsets.end() - 1);
  rmm::device_uvector<char> chars(total_length, stream);

  // We generate the strings in parallel into the `chars` vector using the
  // offsets vector generated above.
  thrust::for_each_n(rmm::exec_policy(stream),
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

/**
 * @brief Generate a column of random numbers
 * @param lower The lower bound of the random numbers
 * @param upper The upper bound of the random numbers
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_rand_num_col(T const& lower,
                                               T const& upper,
                                               cudf::size_type const& num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  cudf::data_type type = get_cudf_type<T>();
  auto col = cudf::make_numeric_column(type, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::size_type begin = 0;
  cudf::size_type end   = num_rows;
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(begin),
                    thrust::make_counting_iterator(end),
                    col->mutable_view().begin<T>(),
                    gen_rand_num<T>(lower, upper));
  return col;
}

/**
 * @brief Generate a primary key column
 *
 * @param start The starting value of the primary key
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_primary_key_col(T const& start,
                                                  cudf::size_type const& num_rows,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const init = cudf::numeric_scalar<T>(start);
  auto const step = cudf::numeric_scalar<T>(1);
  return cudf::sequence(num_rows, init, step, stream, mr);
}

/**
 * @brief Generate a column where all the rows have the same string value
 *
 * @param value The string value to fill the column with
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rep_str_col(std::string const& value,
                                              cudf::size_type const& num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const indices = rmm::device_uvector<cudf::string_view>(num_rows, stream);
  auto const empty_str_col =
    cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), stream, mr);
  auto const scalar = cudf::string_scalar(value);
  return cudf::fill(empty_str_col->view(), 0, num_rows, scalar, stream, mr);
}

/**
 * @brief Generate a column by randomly choosing from set of strings
 *
 * @param set The set of strings to choose from
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_rand_str_col_from_set(std::vector<std::string> set,
                                                        cudf::size_type const& num_rows,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Build a gather map of random strings to choose from
  // The size of the string sets always fits within 16-bit integers
  auto const keys       = gen_primary_key_col<int16_t>(0, set.size(), stream, mr);
  auto const values     = cudf::test::strings_column_wrapper(set.begin(), set.end()).release();
  auto const gather_map = cudf::table_view({keys->view(), values->view()});

  // Build a column of random keys to gather from the set
  auto const indices = gen_rand_num_col<int16_t>(0, set.size() - 1, num_rows, stream, mr);

  // Perform the gather operation
  auto const gathered_table =
    cudf::gather(gather_map, indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::make_unique<cudf::column>(gathered_table->get_column(1));
}

/**
 * @brief Generate a phone number column according to TPC-H specification clause 4.2.2.9
 *
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::unique_ptr<cudf::column> gen_phone_col(cudf::size_type const& num_rows,
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

/**
 * @brief Generate a column consisting of a repeating sequence of integers
 *
 * @param limit The upper limit of the repeating sequence
 * @param num_rows The number of rows in the column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename T>
std::unique_ptr<cudf::column> gen_rep_seq_col(T const& seq_length,
                                              bool zero_indexed,
                                              cudf::size_type const& num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto pkey                    = gen_primary_key_col<cudf::size_type>(0, num_rows, stream, mr);
  auto repeat_seq_zero_indexed = cudf::binary_operation(pkey->view(),
                                                        cudf::numeric_scalar<T>(seq_length),
                                                        cudf::binary_operator::MOD,
                                                        get_cudf_type<T>(),
                                                        stream,
                                                        mr);
  if (zero_indexed) { return repeat_seq_zero_indexed; }
  return cudf::binary_operation(repeat_seq_zero_indexed->view(),
                                cudf::numeric_scalar<T>(1),
                                cudf::binary_operator::ADD,
                                get_cudf_type<T>(),
                                stream,
                                mr);
}

std::unique_ptr<cudf::column> gen_addr_col(cudf::size_type const& num_rows,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return gen_rand_str_col(10, 40, num_rows, stream, mr);
}

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
