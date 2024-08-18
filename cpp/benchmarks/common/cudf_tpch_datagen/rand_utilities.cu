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

#include "rand_utilities.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/filling.hpp>
#include <cudf/strings/detail/strings_children.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>

#include <string>
#include <vector>

namespace {

// Functor for generating random strings
struct gen_rand_str {
  char* chars;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;

  __host__ __device__ gen_rand_str(char* c) : chars(c), char_dist(44, 122) {}

  __host__ __device__ void operator()(thrust::tuple<int64_t, int64_t> str_begin_end)
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

}  // namespace

std::unique_ptr<cudf::column> gen_rand_str_col(cudf::size_type const& lower,
                                               cudf::size_type const& upper,
                                               cudf::size_type const& num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto offsets_begin =
    cudf::detail::make_counting_transform_iterator(0, gen_rand_num<cudf::size_type>(lower, upper));
  auto [offsets_column, computed_bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_begin, offsets_begin + num_rows, stream, mr);
  rmm::device_uvector<char> chars(computed_bytes, stream);

  auto const offset_itr =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // We generate the strings in parallel into the `chars` vector using the
  // offsets vector generated above.
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_zip_iterator(offset_itr, offset_itr + 1),
                     num_rows,
                     gen_rand_str(chars.data()));

  return cudf::make_strings_column(
    num_rows, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

template <typename T>
std::unique_ptr<cudf::column> gen_rand_num_col(T const& lower,
                                               T const& upper,
                                               cudf::size_type const& num_rows,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_to_id<T>()}, num_rows, cudf::mask_state::UNALLOCATED, stream, mr);
  cudf::size_type begin = 0;
  cudf::size_type end   = num_rows;
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(begin),
                    thrust::make_counting_iterator(end),
                    col->mutable_view().begin<T>(),
                    gen_rand_num<T>(lower, upper));
  return col;
}

template std::unique_ptr<cudf::column> gen_rand_num_col<int8_t>(int8_t const& lower,
                                                                int8_t const& upper,
                                                                cudf::size_type const& num_rows,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> gen_rand_num_col<int16_t>(int16_t const& lower,
                                                                 int16_t const& upper,
                                                                 cudf::size_type const& num_rows,
                                                                 rmm::cuda_stream_view stream,
                                                                 rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> gen_rand_num_col<cudf::size_type>(
  cudf::size_type const& lower,
  cudf::size_type const& upper,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> gen_rand_num_col<double>(double const& lower,
                                                                double const& upper,
                                                                cudf::size_type const& num_rows,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> gen_primary_key_col(cudf::scalar const& start,
                                                  cudf::size_type const& num_rows,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return cudf::sequence(num_rows, start, stream, mr);
}

std::unique_ptr<cudf::column> gen_rep_str_col(std::string const& value,
                                              cudf::size_type const& num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const scalar = cudf::string_scalar(value);
  return cudf::make_column_from_scalar(scalar, num_rows, stream, mr);
}

std::unique_ptr<cudf::column> gen_rand_str_col_from_set(std::vector<std::string> set,
                                                        cudf::size_type const& num_rows,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // Build a gather map of random strings to choose from
  // The size of the string sets always fits within 16-bit integers
  auto const keys   = gen_primary_key_col(cudf::numeric_scalar<int16_t>(0), set.size(), stream, mr);
  auto const values = cudf::test::strings_column_wrapper(set.begin(), set.end()).release();
  auto const gather_map = cudf::table_view({keys->view(), values->view()});

  // Build a column of random keys to gather from the set
  auto const indices = gen_rand_num_col<int16_t>(0, set.size() - 1, num_rows, stream, mr);

  // Perform the gather operation
  auto const gathered_table =
    cudf::gather(gather_map, indices->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  return std::make_unique<cudf::column>(gathered_table->get_column(1));
}

template <typename T>
std::unique_ptr<cudf::column> gen_rep_seq_col(T const& seq_length,
                                              bool zero_indexed,
                                              cudf::size_type const& num_rows,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto pkey = gen_primary_key_col(cudf::numeric_scalar<cudf::size_type>(0), num_rows, stream, mr);
  auto repeat_seq_zero_indexed = cudf::binary_operation(pkey->view(),
                                                        cudf::numeric_scalar<T>(seq_length),
                                                        cudf::binary_operator::MOD,
                                                        cudf::data_type{cudf::type_to_id<T>()},
                                                        stream,
                                                        mr);
  if (zero_indexed) { return repeat_seq_zero_indexed; }
  return cudf::binary_operation(repeat_seq_zero_indexed->view(),
                                cudf::numeric_scalar<T>(1),
                                cudf::binary_operator::ADD,
                                cudf::data_type{cudf::type_to_id<T>()},
                                stream,
                                mr);
}

template std::unique_ptr<cudf::column> gen_rep_seq_col<int8_t>(int8_t const& seq_length,
                                                               bool zero_indexed,
                                                               cudf::size_type const& num_rows,
                                                               rmm::cuda_stream_view stream,
                                                               rmm::device_async_resource_ref mr);

template std::unique_ptr<cudf::column> gen_rep_seq_col<cudf::size_type>(
  cudf::size_type const& seq_length,
  bool zero_indexed,
  cudf::size_type const& num_rows,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
