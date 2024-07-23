/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/io/parquet.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <src/io/utilities/multibuffer_memset.hpp>

#include <type_traits>

template <typename T>
struct MultiBufferTestIntegral : public cudf::test::BaseFixture {};

TEST(MultiBufferTestIntegral, BasicTest1)
{
  std::vector<long> BUF_SIZES{5000};
  long NUM_BUFS = BUF_SIZES.size();

  // Device init
  auto stream = cudf::get_default_stream();
  auto _mr    = rmm::mr::get_current_device_resource();

  // Init Host buffers to 0xEE and 0xFF as padding around it
  std::vector<uint64_t*> host_bufs(NUM_BUFS, nullptr);
  for (int i = 0; i < NUM_BUFS; i++) {
    CUDF_CUDA_TRY(cudaMalloc(&host_bufs[i], (BUF_SIZES[i] + 2000) * 8));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i], 0xFF, (BUF_SIZES[i] + 2000) * 8, stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i] + 1000, 0xEE, BUF_SIZES[i] * 8, stream.value()));
  }

  // Init Device buffers
  std::vector<cudf::device_span<uint64_t>> device_bufs(NUM_BUFS);
  std::transform(thrust::make_counting_iterator(0L),
                 thrust::make_counting_iterator(NUM_BUFS),
                 device_bufs.begin(),
                 [BUF_SIZES, host_bufs](auto i) {
                   return cudf::device_span<uint64_t>(host_bufs[i] + 1000, BUF_SIZES[i]);
                 });

  // Function Call
  multibuffer_memset(device_bufs, 0UL, stream, _mr);

  // Compare to see that only given buffers are zeroed out
  std::for_each(thrust::make_counting_iterator(0L),
                thrust::make_counting_iterator(NUM_BUFS),
                [BUF_SIZES, host_bufs, device_bufs](auto i) {
                  std::vector<uint64_t> temp(BUF_SIZES[i] + 2000);
                  cudf::host_span<uint64_t> host(temp);
                  CUDF_CUDA_TRY(cudaMemcpy(
                    host.data(), host_bufs[i], (BUF_SIZES[i] + 2000) * 8, cudaMemcpyDefault));
                  for (int j = 0; j < BUF_SIZES[i] + 2000; j++) {
                    if (j < 1000 || j >= BUF_SIZES[i] + 1000) {
                      EXPECT_EQ(host[j], 0xFFFFFFFFFFFFFFFF);
                    } else {
                      EXPECT_EQ(host[j], 0UL);
                    }
                  }
                });
}

TEST(MultiBufferTestIntegral, BasicTest2)
{
  std::vector<long> BUF_SIZES{50000, 4, 1000, 250000, 1, 100, 8000};
  long NUM_BUFS = BUF_SIZES.size();

  // Device init
  auto stream = cudf::get_default_stream();
  auto _mr    = rmm::mr::get_current_device_resource();

  // Init Host buffers to 0xEE and 0xFF as padding around it
  std::vector<uint64_t*> host_bufs(NUM_BUFS, nullptr);
  for (int i = 0; i < NUM_BUFS; i++) {
    CUDF_CUDA_TRY(cudaMalloc(&host_bufs[i], (BUF_SIZES[i] + 2000) * 8));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i], 0xFF, (BUF_SIZES[i] + 2000) * 8, stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i] + 1000, 0xEE, BUF_SIZES[i] * 8, stream.value()));
  }

  // Init Device buffers
  std::vector<cudf::device_span<uint64_t>> device_bufs(NUM_BUFS);
  std::transform(thrust::make_counting_iterator(0L),
                 thrust::make_counting_iterator(NUM_BUFS),
                 device_bufs.begin(),
                 [BUF_SIZES, host_bufs](auto i) {
                   return cudf::device_span<uint64_t>(host_bufs[i] + 1000, BUF_SIZES[i]);
                 });

  // Function Call
  multibuffer_memset(device_bufs, 0UL, stream, _mr);

  // Compare to see that only given buffers are zeroed out
  std::for_each(thrust::make_counting_iterator(0L),
                thrust::make_counting_iterator(NUM_BUFS),
                [BUF_SIZES, host_bufs, device_bufs](auto i) {
                  std::vector<uint64_t> temp(BUF_SIZES[i] + 2000);
                  cudf::host_span<uint64_t> host(temp);
                  CUDF_CUDA_TRY(cudaMemcpy(
                    host.data(), host_bufs[i], (BUF_SIZES[i] + 2000) * 8, cudaMemcpyDefault));
                  for (int j = 0; j < BUF_SIZES[i] + 2000; j++) {
                    if (j < 1000 || j >= BUF_SIZES[i] + 1000) {
                      EXPECT_EQ(host[j], 0xFFFFFFFFFFFFFFFF);
                    } else {
                      EXPECT_EQ(host[j], 0UL);
                    }
                  }
                });
}

TEST(MultiBufferTestIntegral, BasicTest3)
{
  std::vector<long> BUF_SIZES{0, 1, 100, 1000, 10000, 100000, 0, 1, 100000};
  long NUM_BUFS = BUF_SIZES.size();

  // Device init
  auto stream = cudf::get_default_stream();
  auto _mr    = rmm::mr::get_current_device_resource();

  // Init Host buffers to 0xEE and 0xFF as padding around it
  std::vector<uint64_t*> host_bufs(NUM_BUFS, nullptr);
  for (int i = 0; i < NUM_BUFS; i++) {
    CUDF_CUDA_TRY(cudaMalloc(&host_bufs[i], (BUF_SIZES[i] + 2000) * 8));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i], 0xFF, (BUF_SIZES[i] + 2000) * 8, stream.value()));
    CUDF_CUDA_TRY(cudaMemsetAsync(host_bufs[i] + 1000, 0xEE, BUF_SIZES[i] * 8, stream.value()));
  }

  // Init Device buffers
  std::vector<cudf::device_span<uint64_t>> device_bufs(NUM_BUFS);
  std::transform(thrust::make_counting_iterator(0L),
                 thrust::make_counting_iterator(NUM_BUFS),
                 device_bufs.begin(),
                 [BUF_SIZES, host_bufs](auto i) {
                   return cudf::device_span<uint64_t>(host_bufs[i] + 1000, BUF_SIZES[i]);
                 });

  // Function Call
  multibuffer_memset(device_bufs, 0UL, stream, _mr);

  // Compare to see that only given buffers are zeroed out
  std::for_each(thrust::make_counting_iterator(0L),
                thrust::make_counting_iterator(NUM_BUFS),
                [BUF_SIZES, host_bufs, device_bufs](auto i) {
                  std::vector<uint64_t> temp(BUF_SIZES[i] + 2000);
                  cudf::host_span<uint64_t> host(temp);
                  CUDF_CUDA_TRY(cudaMemcpy(
                    host.data(), host_bufs[i], (BUF_SIZES[i] + 2000) * 8, cudaMemcpyDefault));
                  for (int j = 0; j < BUF_SIZES[i] + 2000; j++) {
                    if (j < 1000 || j >= BUF_SIZES[i] + 1000) {
                      EXPECT_EQ(host[j], 0xFFFFFFFFFFFFFFFF);
                    } else {
                      EXPECT_EQ(host[j], 0UL);
                    }
                  }
                });
}
