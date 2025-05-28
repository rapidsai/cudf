/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <type_traits>

template <typename T>
struct MultiBufferTestIntegral : public cudf::test::BaseFixture {};

TEST(MultiBufferTestIntegral, BasicTest1)
{
  std::vector<size_t> const buffer_sizes{
    50000, 4, 1000, 0, 250000, 1, 100, 8000, 0, 1, 100, 1000, 10000, 100000, 0, 1, 100000};

  // Device init
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Creating base vector for data and setting it to all 0xFF
  std::vector<std::vector<uint64_t>> expected;
  std::transform(
    buffer_sizes.begin(), buffer_sizes.end(), std::back_inserter(expected), [](auto size) {
      return std::vector<uint64_t>(size + 2000, std::numeric_limits<uint64_t>::max());
    });

  // set buffer region to other value
  std::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(expected.begin(), buffer_sizes.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected.end(), buffer_sizes.end())),
    [](auto elem) {
      std::fill_n(thrust::get<0>(elem).begin() + 1000, thrust::get<1>(elem), 0xEEEEEEEEEEEEEEEE);
    });

  // Copy host vector data to device
  std::vector<rmm::device_uvector<uint64_t>> device_buffers;
  std::transform(expected.begin(),
                 expected.end(),
                 std::back_inserter(device_buffers),
                 [stream, mr](auto const& vec) {
                   return cudf::detail::make_device_uvector_async(vec, stream, mr);
                 });

  // Initialize device buffers for memset
  auto buffers =
    cudf::detail::make_host_vector<cudf::device_span<uint64_t>>(device_buffers.size(), stream);
  std::transform(
    thrust::make_zip_iterator(thrust::make_tuple(device_buffers.begin(), buffer_sizes.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(device_buffers.end(), buffer_sizes.end())),
    buffers.begin(),
    [](auto const& elem) {
      return cudf::device_span<uint64_t>(thrust::get<0>(elem).data() + 1000, thrust::get<1>(elem));
    });

  // Function call
  cudf::detail::batched_memset<uint64_t>(buffers, uint64_t{0}, stream);

  // Set all buffer regions to 0 for expected comparison
  std::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(expected.begin(), buffer_sizes.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(expected.end(), buffer_sizes.end())),
    [](auto elem) { std::fill_n(thrust::get<0>(elem).begin() + 1000, thrust::get<1>(elem), 0UL); });

  // Compare to see that only given buffers are zeroed out
  std::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(device_buffers.begin(), expected.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(device_buffers.end(), expected.end())),
    [stream](auto const& elem) {
      auto const after_memset = cudf::detail::make_host_vector(thrust::get<0>(elem), stream);
      EXPECT_TRUE(
        std::equal(thrust::get<1>(elem).begin(), thrust::get<1>(elem).end(), after_memset.begin()));
    });
}
