/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <tests/strings/utilities.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cstring>
#include <vector>

struct StringsFactoriesTest : public cudf::test::BaseFixture {
};

TEST_F(StringsFactoriesTest, CreateColumnFromPair)
{
  std::vector<const char*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  thrust::host_vector<char> h_buffer(memsize);
  rmm::device_uvector<char> d_buffer(memsize, rmm::cuda_stream_default);
  thrust::host_vector<thrust::pair<const char*, cudf::size_type>> strings(count);
  thrust::host_vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset = 0;
  cudf::size_type nulls  = 0;
  h_offsets[0]           = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    const char* str = h_test_strings[idx];
    if (!str) {
      strings[idx] = thrust::pair<const char*, cudf::size_type>{nullptr, 0};
      nulls++;
    } else {
      cudf::size_type length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      strings[idx] = thrust::pair<const char*, cudf::size_type>{d_buffer.data() + offset, length};
      offset += length;
    }
    h_offsets[idx + 1] = offset;
  }
  auto d_strings = cudf::detail::make_device_uvector_sync(strings);
  CUDA_TRY(cudaMemcpy(d_buffer.data(), h_buffer.data(), memsize, cudaMemcpyHostToDevice));
  auto column = cudf::make_strings_column(d_strings);
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), nulls);
  if (nulls) {
    EXPECT_TRUE(column->nullable());
    EXPECT_TRUE(column->has_nulls());
  }
  EXPECT_EQ(2, column->num_children());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars().size(), memsize);

  // check string data
  auto h_chars_data = cudf::detail::make_std_vector_sync(
    cudf::device_span<char const>(strings_view.chars().data<char>(), strings_view.chars().size()),
    rmm::cuda_stream_default);
  auto h_offsets_data = cudf::detail::make_std_vector_sync(
    cudf::device_span<cudf::offset_type const>(
      strings_view.offsets().data<cudf::offset_type>() + strings_view.offset(),
      strings_view.size() + 1),
    rmm::cuda_stream_default);
  EXPECT_EQ(memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0);
  EXPECT_EQ(
    memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size() * sizeof(cudf::size_type)), 0);
}

TEST_F(StringsFactoriesTest, CreateColumnFromOffsets)
{
  std::vector<const char*> h_test_strings{"the quick brown fox jumps over the lazy dog",
                                          "the fat cat lays next to the other accénted cat",
                                          "a slow moving turtlé cannot catch the bird",
                                          "which can be composéd together to form a more complete",
                                          "thé result does not include the value in the sum in",
                                          "",
                                          nullptr,
                                          "absent stop words"};

  cudf::size_type memsize = 0;
  for (auto itr = h_test_strings.begin(); itr != h_test_strings.end(); ++itr)
    memsize += *itr ? (cudf::size_type)strlen(*itr) : 0;
  cudf::size_type count = (cudf::size_type)h_test_strings.size();
  std::vector<char> h_buffer(memsize);
  std::vector<cudf::size_type> h_offsets(count + 1);
  cudf::size_type offset         = 0;
  h_offsets[0]                   = offset;
  cudf::bitmask_type h_null_mask = 0;
  cudf::size_type null_count     = 0;
  for (cudf::size_type idx = 0; idx < count; ++idx) {
    h_null_mask     = (h_null_mask << 1);
    const char* str = h_test_strings[idx];
    if (str) {
      cudf::size_type length = (cudf::size_type)strlen(str);
      memcpy(h_buffer.data() + offset, str, length);
      offset += length;
      h_null_mask |= 1;
    } else
      null_count++;
    h_offsets[idx + 1] = offset;
  }

  std::vector<cudf::bitmask_type> h_nulls{h_null_mask};
  auto d_buffer  = cudf::detail::make_device_uvector_sync(h_buffer);
  auto d_offsets = cudf::detail::make_device_uvector_sync(h_offsets);
  auto d_nulls   = cudf::detail::make_device_uvector_sync(h_nulls);
  auto column    = cudf::make_strings_column(d_buffer, d_offsets, d_nulls, null_count);
  EXPECT_EQ(column->type(), cudf::data_type{cudf::type_id::STRING});
  EXPECT_EQ(column->null_count(), null_count);
  EXPECT_EQ(2, column->num_children());

  cudf::strings_column_view strings_view(column->view());
  EXPECT_EQ(strings_view.size(), count);
  EXPECT_EQ(strings_view.offsets().size(), count + 1);
  EXPECT_EQ(strings_view.chars().size(), memsize);

  // check string data
  auto h_chars_data = cudf::detail::make_std_vector_sync(
    cudf::device_span<char const>(strings_view.chars().data<char>(), strings_view.chars().size()),
    rmm::cuda_stream_default);
  auto h_offsets_data = cudf::detail::make_std_vector_sync(
    cudf::device_span<cudf::offset_type const>(
      strings_view.offsets().data<cudf::offset_type>() + strings_view.offset(),
      strings_view.size() + 1),
    rmm::cuda_stream_default);
  EXPECT_EQ(memcmp(h_buffer.data(), h_chars_data.data(), h_buffer.size()), 0);
  EXPECT_EQ(
    memcmp(h_offsets.data(), h_offsets_data.data(), h_offsets.size() * sizeof(cudf::size_type)), 0);
}

TEST_F(StringsFactoriesTest, CreateScalar)
{
  std::string value = "test string";
  auto s            = cudf::make_string_scalar(value);
  auto string_s     = static_cast<cudf::string_scalar*>(s.get());

  EXPECT_EQ(string_s->to_string(), value);
  EXPECT_TRUE(string_s->is_valid());
  EXPECT_TRUE(s->is_valid());
}

TEST_F(StringsFactoriesTest, EmptyStringsColumn)
{
  rmm::device_uvector<char> d_chars{0, rmm::cuda_stream_default};
  auto d_offsets = cudf::detail::make_zeroed_device_uvector_sync<cudf::size_type>(1);
  rmm::device_uvector<cudf::bitmask_type> d_nulls{0, rmm::cuda_stream_default};

  auto results = cudf::make_strings_column(d_chars, d_offsets, d_nulls, 0);
  cudf::test::expect_strings_empty(results->view());

  rmm::device_uvector<thrust::pair<const char*, cudf::size_type>> d_strings{
    0, rmm::cuda_stream_default};
  results = cudf::make_strings_column(d_strings);
  cudf::test::expect_strings_empty(results->view());
}

namespace {
using string_pair = thrust::pair<char const*, cudf::size_type>;
struct string_view_to_pair {
  __device__ string_pair operator()(thrust::pair<cudf::string_view, bool> const& p)
  {
    return (p.second) ? string_pair{p.first.data(), p.first.size_bytes()} : string_pair{nullptr, 0};
  }
};
}  // namespace

TEST_F(StringsFactoriesTest, StringPairWithNullsAndEmpty)
{
  cudf::test::strings_column_wrapper data(
    {"", "this", "is", "", "a", "", "column", "of", "strings", "", ""},
    {0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1});

  auto d_column = cudf::column_device_view::create(data);
  rmm::device_uvector<string_pair> pairs(d_column->size(), rmm::cuda_stream_default);
  thrust::transform(thrust::device,
                    d_column->pair_begin<cudf::string_view, true>(),
                    d_column->pair_end<cudf::string_view, true>(),
                    pairs.data(),
                    string_view_to_pair{});

  auto result = cudf::make_strings_column(pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(result->view(), data);
}
