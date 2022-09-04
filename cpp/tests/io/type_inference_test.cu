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

#include <io/utilities/hostdevice_vector.hpp>
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
  auto stream  = rmm::cuda_stream_default;
  auto options = inference_options{};

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

  auto res_type = detect_data_type(options.view(), d_data, d_col_strings, size, 0, stream);

  EXPECT_EQ(res_type, cudf::data_type{cudf::type_id::INT64});
}
