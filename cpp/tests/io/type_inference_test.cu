/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <io/utilities/trie.cuh>
#include <io/utilities/type_inference.cuh>

#include <cudf_test/base_fixture.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <cstddef>
#include <string>
#include <vector>

using cudf::io::detail::detect_data_type;
using cudf::io::detail::inference_options;

// Base test fixture for tests
struct TypeInference : public cudf::test::BaseFixture {
};

TEST_F(TypeInference, Basic)
{
  auto const stream = rmm::cuda_stream_default;
  auto options      = inference_options{};

  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data = "[42,52,5]";
  rmm::device_uvector<char> d_data{data.size(), stream};
  cudaMemcpyAsync(
    d_data.data(), data.data(), data.size() * sizeof(char), cudaMemcpyHostToDevice, stream.value());

  std::size_t constexpr size = 3;
  auto const string_offset   = std::vector<int32_t>{1, 4, 7};
  auto const string_length   = std::vector<std::size_t>{2, 2, 1};
  rmm::device_vector<int32_t> d_string_offset{string_offset};
  rmm::device_vector<std::size_t> d_string_length{string_length};

  auto d_col_strings =
    thrust::make_zip_iterator(make_tuple(d_string_offset.begin(), d_string_length.begin()));

  cudf::size_type constexpr num_omitted_nulls = 0;
  auto res_type =
    detect_data_type(options.view(), d_data, d_col_strings, num_omitted_nulls, size, stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::INT64});
}

TEST_F(TypeInference, OmittedNull)
{
  auto const stream = rmm::cuda_stream_default;
  auto options      = inference_options{};

  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data = "[52,5]";
  rmm::device_uvector<char> d_data{data.size(), stream};
  cudaMemcpyAsync(
    d_data.data(), data.data(), data.size() * sizeof(char), cudaMemcpyHostToDevice, stream.value());

  std::size_t constexpr size = 3;
  auto const string_offset   = std::vector<int32_t>{1, 1, 4};
  auto const string_length   = std::vector<std::size_t>{0, 2, 1};
  rmm::device_vector<int32_t> d_string_offset{string_offset};
  rmm::device_vector<std::size_t> d_string_length{string_length};

  auto d_col_strings =
    thrust::make_zip_iterator(make_tuple(d_string_offset.begin(), d_string_length.begin()));

  cudf::size_type constexpr num_omitted_nulls = 1;
  auto res_type =
    detect_data_type(options.view(), d_data, d_col_strings, num_omitted_nulls, size, stream);

  EXPECT_EQ(res_type,
            cudf::data_type{cudf::type_id::FLOAT64});  // FLOAT64 to align with pandas's behavior
}

TEST_F(TypeInference, String)
{
  auto const stream = rmm::cuda_stream_default;
  auto options      = inference_options{};

  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data = "[\"1990\",\"8\",\"25\"]";
  rmm::device_uvector<char> d_data{data.size(), stream};
  cudaMemcpyAsync(
    d_data.data(), data.data(), data.size() * sizeof(char), cudaMemcpyHostToDevice, stream.value());

  std::size_t constexpr size = 3;
  auto const string_offset   = std::vector<int32_t>{1, 8, 12};
  auto const string_length   = std::vector<std::size_t>{6, 3, 4};
  rmm::device_vector<int32_t> d_string_offset{string_offset};
  rmm::device_vector<std::size_t> d_string_length{string_length};

  auto d_col_strings =
    thrust::make_zip_iterator(make_tuple(d_string_offset.begin(), d_string_length.begin()));

  cudf::size_type constexpr num_omitted_nulls = 0;
  auto res_type =
    detect_data_type(options.view(), d_data, d_col_strings, num_omitted_nulls, size, stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::STRING});
}

TEST_F(TypeInference, Bool)
{
  auto const stream = rmm::cuda_stream_default;
  auto options      = inference_options{};

  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data = "[true,false,false]";
  rmm::device_uvector<char> d_data{data.size(), stream};
  cudaMemcpyAsync(
    d_data.data(), data.data(), data.size() * sizeof(char), cudaMemcpyHostToDevice, stream.value());

  std::size_t constexpr size = 3;
  auto const string_offset   = std::vector<int32_t>{1, 6, 12};
  auto const string_length   = std::vector<std::size_t>{4, 5, 5};
  rmm::device_vector<int32_t> d_string_offset{string_offset};
  rmm::device_vector<std::size_t> d_string_length{string_length};

  auto d_col_strings =
    thrust::make_zip_iterator(make_tuple(d_string_offset.begin(), d_string_length.begin()));

  cudf::size_type constexpr num_omitted_nulls = 0;
  auto res_type =
    detect_data_type(options.view(), d_data, d_col_strings, num_omitted_nulls, size, stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::BOOL8});
}

TEST_F(TypeInference, Timestamp)
{
  auto const stream = rmm::cuda_stream_default;
  auto options      = inference_options{};

  options.trie_true  = cudf::detail::create_serialized_trie({"true"}, stream);
  options.trie_false = cudf::detail::create_serialized_trie({"false"}, stream);
  options.trie_na    = cudf::detail::create_serialized_trie({"", "null"}, stream);

  std::string data = "[1970/2/5,1970/8/25]";
  rmm::device_uvector<char> d_data{data.size(), stream};
  cudaMemcpyAsync(
    d_data.data(), data.data(), data.size() * sizeof(char), cudaMemcpyHostToDevice, stream.value());

  std::size_t constexpr size = 3;
  auto const string_offset   = std::vector<int32_t>{1, 10};
  auto const string_length   = std::vector<std::size_t>{8, 9};
  rmm::device_vector<int32_t> d_string_offset{string_offset};
  rmm::device_vector<std::size_t> d_string_length{string_length};

  auto d_col_strings =
    thrust::make_zip_iterator(make_tuple(d_string_offset.begin(), d_string_length.begin()));

  cudf::size_type constexpr num_omitted_nulls = 0;
  auto res_type =
    detect_data_type(options.view(), d_data, d_col_strings, num_omitted_nulls, size, stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::TIMESTAMP_MILLISECONDS});
}
