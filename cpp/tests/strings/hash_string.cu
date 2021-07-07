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

#include "./utilities.h"
#include "rmm/exec_policy.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>
#include <vector>

struct StringsHashTest : public cudf::test::BaseFixture {
};

struct hash_string_fn {
  cudf::column_device_view d_strings;
  uint32_t __device__ operator()(uint32_t idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto item = d_strings.element<cudf::string_view>(idx);
    return MurmurHash3_32<cudf::string_view>{}(item);
  }
};

TEST_F(StringsHashTest, HashTest)
{
  std::vector<const char*> h_strings{"abcdefghijklmnopqrstuvwxyz",
                                     "abcdefghijklmnopqrstuvwxyz",
                                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                     "0123456789",
                                     "4",
                                     "",
                                     nullptr,
                                     "last one"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto strings_view   = cudf::strings_column_view(strings);
  auto strings_column = cudf::column_device_view::create(strings_view.parent());
  auto d_view         = *strings_column;

  rmm::device_uvector<uint32_t> d_values(strings_view.size(), rmm::cuda_stream_default);
  thrust::transform(rmm::exec_policy(),
                    thrust::make_counting_iterator<uint32_t>(0),
                    thrust::make_counting_iterator<uint32_t>(strings_view.size()),
                    d_values.begin(),
                    hash_string_fn{d_view});

  uint32_t h_expected[] = {
    2739798893, 2739798893, 3506676360, 1891213601, 3778137224, 0, 0, 1551088011};
  auto h_values = cudf::detail::make_host_vector_sync(d_values);
  for (uint32_t idx = 0; idx < h_values.size(); ++idx)
    EXPECT_EQ(h_values[idx], h_expected[idx]);
}
