/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/common/generate_input.hpp>

#include <cudf/detail/null_mask.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

namespace {

constexpr float null_probability = 0.5;

struct anding {
  __device__ cudf::bitmask_type operator()(cudf::bitmask_type left, cudf::bitmask_type right) { 
    return left & right; 
  }
};

void BM_segmented_bitmask_and(nvbench::state& state)
{
  unsigned seed = 12345;
  
  // Get benchmark parameters
  auto const num_segments = static_cast<size_t>(state.get_int64("num_segments"));
  auto const expected_masks_per_segment = static_cast<size_t>(state.get_int64("expected_masks_per_segment"));
  auto const mask_size_bits = static_cast<size_t>(state.get_int64("mask_size_bits"));

  std::mt19937 generator(seed);
  std::normal_distribution normal_dist(static_cast<double>(expected_masks_per_segment), 1.0);
  std::vector<cudf::size_type> segments(num_segments + 1);
  auto num_masks = 0;
  std::generate_n(segments.begin(), num_segments, [&normal_dist, &generator, &num_masks]() {
    cudf::size_type segment_size = normal_dist(generator);
    num_masks += segment_size;
    return segment_size;
  });
  std::exclusive_scan(segments.begin(), segments.end(), segments.begin(), 0);

  std::vector<rmm::device_buffer> masks;
  std::vector<cudf::bitmask_type*> mask_pointers;
  masks.reserve(num_masks);
  std::generate_n(std::back_inserter(masks), num_masks, [mask_size_bits, seed, &mask_pointers]() {
    auto mask_pair = create_random_null_mask(mask_size_bits, null_probability, seed);
    mask_pointers.push_back(static_cast<cudf::bitmask_type*>(mask_pair.first.data()));
    return std::move(mask_pair.first);
  });

  std::vector<cudf::size_type> mask_begin_bits(num_masks, 0);

  auto const data_bytes = num_masks * std::ceil(static_cast<double>(mask_size_bits) / 8) + (sizeof(cudf::size_type) * (num_masks + num_segments));
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(data_bytes, "input size");
  state.template add_global_memory_reads<nvbench::int8_t>(data_bytes);
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               auto result = cudf::detail::segmented_bitmask_binop(
                 anding{},
                 mask_pointers,
                 mask_begin_bits,
                 mask_size_bits,
                 segments, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
               timer.stop();
             });
  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count((static_cast<double>(data_bytes) / (1024 * 1024)) / time,
                          "Mbytes_per_second");
}

} // anonymous namespace
  
NVBENCH_BENCH(BM_segmented_bitmask_and)
  .set_name("segmented_bitmask_and")
  .add_int64_axis("num_segments", {10, 100, 1000, 10000})
  .add_int64_axis("expected_masks_per_segment", {4, 8})
  .add_int64_axis("mask_size_bits", {64, 128, 512, 1024});
