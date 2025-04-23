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

#include <cudf/null_mask.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <random>

namespace {

std::vector<cudf::size_type> generate_end_bits(cudf::size_type num_masks,
                                               cudf::size_type mask_size,
                                               bool generate_random_sizes)
{
  std::vector<cudf::size_type> end_bits(num_masks);
  if (generate_random_sizes) {
    std::default_random_engine generator;
    std::uniform_int_distribution<cudf::size_type> dist(1, mask_size);
    std::generate(end_bits.begin(), end_bits.end(), [&]() { return dist(generator); });
  } else {
    std::fill(end_bits.begin(), end_bits.end(), mask_size);
  }
  return end_bits;
}

}  // namespace

void BM_setnullmask(nvbench::state& state)
{
  auto const mask_size    = static_cast<cudf::size_type>(state.get_int64("mask_size"));
  rmm::device_buffer mask = cudf::create_null_mask(mask_size, cudf::mask_state::UNINITIALIZED);
  auto begin = 0, end = mask_size;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), begin, end, true);
               timer.stop();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count((static_cast<double>(mask_size) / (8 * 1024 * 1024)) / time,
                          "Mbytes_per_second");
}

void BM_setnullmask_bulk(nvbench::state& state)
{
  srand(31337);

  auto const mask_size = static_cast<cudf::size_type>(state.get_int64("max_mask_size"));
  auto const num_masks = static_cast<cudf::size_type>(state.get_int64("num_masks"));
  bool const use_variable_mask_sizes = static_cast<bool>(state.get_int64("use_variable_mask_size"));

  std::vector<cudf::size_type> begin_bits(num_masks, 0);
  std::vector<cudf::size_type> end_bits =
    generate_end_bits(num_masks, mask_size, use_variable_mask_sizes);

  auto valids = thrust::host_vector<bool>(num_masks, true);

  std::vector<rmm::device_buffer> masks(num_masks);
  std::vector<cudf::bitmask_type*> masks_ptr(num_masks);
  for (cudf::size_type i = 0; i < num_masks; ++i) {
    masks[i]     = cudf::create_null_mask(mask_size, cudf::mask_state::UNINITIALIZED);
    masks_ptr[i] = static_cast<cudf::bitmask_type*>(masks[i].data());
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               cudf::set_null_masks(masks_ptr, begin_bits, end_bits, valids);
               timer.stop();
             });
  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count((static_cast<double>(mask_size) / (8 * 1024 * 1024)) * num_masks / time,
                          "Mbytes_per_second");
}

void BM_setnullmask_loop(nvbench::state& state)
{
  srand(31337);

  auto const mask_size = static_cast<cudf::size_type>(state.get_int64("max_mask_size"));
  auto const num_masks = static_cast<cudf::size_type>(state.get_int64("num_masks"));
  bool const use_variable_mask_sizes = static_cast<bool>(state.get_int64("use_variable_mask_size"));

  std::vector<cudf::size_type> begin_bits(num_masks, 0);
  std::vector<cudf::size_type> end_bits =
    generate_end_bits(num_masks, mask_size, use_variable_mask_sizes);

  auto valids = thrust::host_vector<bool>(num_masks, true);

  std::vector<rmm::device_buffer> masks(num_masks);
  std::vector<cudf::bitmask_type*> masks_ptr(num_masks);
  for (cudf::size_type i = 0; i < num_masks; ++i) {
    masks[i]     = cudf::create_null_mask(mask_size, cudf::mask_state::UNINITIALIZED);
    masks_ptr[i] = static_cast<cudf::bitmask_type*>(masks[i].data());
  }

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               timer.start();
               for (auto i = 0; i < num_masks; ++i) {
                 cudf::set_null_mask(masks_ptr[i], 0, end_bits[i], true);
               }
               timer.stop();
             });
  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count((static_cast<double>(mask_size) / (8 * 1024 * 1024)) * num_masks / time,
                          "Mbytes_per_second");
}

NVBENCH_BENCH(BM_setnullmask)
  .set_name("set_nullmask")
  .set_min_samples(4)
  .add_int64_power_of_two_axis("mask_size", nvbench::range(10, 30, 10));

NVBENCH_BENCH(BM_setnullmask_bulk)
  .set_name("set_nullmasks_bulk")
  .set_min_samples(4)
  .add_int64_power_of_two_axis("max_mask_size", nvbench::range(10, 25, 5))
  .add_int64_power_of_two_axis("num_masks", nvbench::range(4, 12, 4))
  .add_int64_axis("use_variable_mask_size", {0, 1});

NVBENCH_BENCH(BM_setnullmask_loop)
  .set_name("set_nullmasks_loop")
  .set_min_samples(4)
  .add_int64_power_of_two_axis("max_mask_size", nvbench::range(10, 25, 5))
  .add_int64_power_of_two_axis("num_masks", nvbench::range(4, 12, 4))
  .add_int64_axis("use_variable_mask_size", {0, 1});
