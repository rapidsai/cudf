/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/segmented_dynamic_csr.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <vector>

namespace {

using cudf::size_type;

struct never_equal {
  __device__ bool operator()(cuda::std::uint64_t, cuda::std::uint64_t) const { return false; }
};

__global__ void force_dynamic_key_map_collisions(cudf::detail::dynamic_csr_key_map_ref map,
                                                 size_type* results)
{
  auto const row = static_cast<size_type>(threadIdx.x);
  if (row < 5) {
    // Every input has the same hash but the comparator declares every pair unequal. This forces
    // a complete linear-probe collision chain and makes the final insert report full capacity.
    results[row] = map.find_or_insert(17, static_cast<cuda::std::uint64_t>(row), never_equal{});
  }
}

}  // namespace

struct SegmentedDynamicCsrTest : public cudf::test::BaseFixture {};

TEST_F(SegmentedDynamicCsrTest, DynamicKeyMapHandlesCollisionsAndReportsFullCapacity)
{
  auto const stream       = cudf::test::get_default_stream();
  constexpr auto capacity = cuda::std::uint32_t{4};
  rmm::device_uvector<cudf::detail::dynamic_csr_key_slot> entries{capacity, stream};
  rmm::device_uvector<size_type> results{5, stream};
  CUDF_CUDA_TRY(cudaMemsetAsync(entries.data(),
                                0xff,
                                entries.size() * sizeof(cudf::detail::dynamic_csr_key_slot),
                                stream.get()));

  force_dynamic_key_map_collisions<<<1, 5, 0, stream.get()>>>(
    {entries.data(), capacity, capacity - 1}, results.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  auto const host_results = cudf::detail::make_host_vector(
    cudf::device_span<size_type const>{results.data(), results.size()}, stream);
  auto const full_count = std::count(host_results.begin(), host_results.end(), size_type{capacity});
  EXPECT_EQ(full_count, 1);

  std::vector<size_type> inserted;
  inserted.reserve(capacity);
  for (auto const slot : host_results) {
    if (slot != capacity) { inserted.push_back(slot); }
  }
  std::sort(inserted.begin(), inserted.end());
  EXPECT_EQ(inserted, (std::vector<size_type>{0, 1, 2, 3}));
}

TEST_F(SegmentedDynamicCsrTest, KeyMapExhaustionReportsOverflow)
{
  auto const stream = cudf::test::get_default_stream();
  rmm::device_uvector<int> exhausted{1, stream};
  CUDF_CUDA_TRY(cudaMemsetAsync(exhausted.data(), 1, sizeof(int), stream.get()));

  EXPECT_THROW(cudf::detail::throw_if_dynamic_key_map_exhausted(exhausted.data(), stream),
               std::overflow_error);
}
