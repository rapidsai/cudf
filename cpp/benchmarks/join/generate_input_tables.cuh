/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <curand.h>
#include <curand_kernel.h>

#include <cassert>

CUDF_KERNEL void init_curand(curandState* state, int const nstates)
{
  int ithread = cudf::detail::grid_1d::global_thread_id();

  if (ithread < nstates) { curand_init(1234ULL, ithread, 0, state + ithread); }
}

template <typename key_type, typename size_type>
CUDF_KERNEL void init_build_tbl(key_type* const build_tbl,
                                size_type const build_tbl_size,
                                int const multiplicity,
                                curandState* state,
                                int const num_states)
{
  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  assert(start_idx < num_states);

  curandState localState = state[start_idx];

  for (cudf::thread_index_type tidx = start_idx; tidx < build_tbl_size; tidx += stride) {
    auto const idx = static_cast<size_type>(tidx);
    double const x = curand_uniform_double(&localState);

    build_tbl[idx] = static_cast<key_type>(x * (build_tbl_size / multiplicity));
  }

  state[start_idx] = localState;
}

template <typename key_type, typename size_type>
CUDF_KERNEL void init_probe_tbl(key_type* const probe_tbl,
                                size_type const probe_tbl_size,
                                size_type const build_tbl_size,
                                key_type const rand_max,
                                double const selectivity,
                                int const multiplicity,
                                curandState* state,
                                int const num_states)
{
  auto const start_idx = cudf::detail::grid_1d::global_thread_id();
  auto const stride    = cudf::detail::grid_1d::grid_stride();
  assert(start_idx < num_states);

  curandState localState = state[start_idx];

  for (cudf::thread_index_type tidx = start_idx; tidx < probe_tbl_size; tidx += stride) {
    auto const idx = static_cast<size_type>(tidx);
    key_type val;
    double x = curand_uniform_double(&localState);

    if (x <= selectivity) {
      // x <= selectivity means this key in the probe table should be present in the build table, so
      // we pick a key from [0, build_tbl_size / multiplicity]
      x   = curand_uniform_double(&localState);
      val = static_cast<key_type>(x * (build_tbl_size / multiplicity));
    } else {
      // This key in the probe table should not be present in the build table, so we pick a key from
      // [build_tbl_size, rand_max].
      x   = curand_uniform_double(&localState);
      val = static_cast<key_type>(x * (rand_max - build_tbl_size) + build_tbl_size);
    }
    probe_tbl[idx] = val;
  }

  state[start_idx] = localState;
}

/**
 * generate_input_tables generates random integer input tables for database benchmarks.
 *
 * generate_input_tables generates two random integer input tables for database benchmark
 * mainly designed to benchmark join operations. The templates key_type and size_type needed
 * to be builtin integer types (e.g. short, int, longlong) and key_type needs to be signed
 * as the lottery used internally relies on being able to use negative values to mark drawn
 * numbers. The tables need to be preallocated in a memory region accessible by the GPU
 * (e.g. device memory, zero copy memory or unified memory). Each value in the build table
 * will be from [0,rand_max] and if uniq_build_tbl_keys is true it is ensured that each value
 * will be uniq in the build table. Each value in the probe table will be also in the build
 * table with a probability of selectivity and a random number from
 * [0,rand_max] \setminus \{build_tbl\} otherwise.
 *
 * @param[out] build_tbl            The build table to generate. Usually the smaller table used to
 *                                  "build" the hash table in a hash based join implementation.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[out] probe_tbl            The probe table to generate. Usually the larger table used to
 *                                  probe into the hash table created from the build table.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[in] selectivity           probability with which an element of the probe table is
 *                                  present in the build table.
 * @param[in] multiplicity          number of matches for each key.
 */
template <typename key_type, typename size_type>
void generate_input_tables(key_type* const build_tbl,
                           size_type const build_tbl_size,
                           key_type* const probe_tbl,
                           size_type const probe_tbl_size,
                           double const selectivity,
                           int const multiplicity)
{
  // With large values of rand_max the a lot of temporary storage is needed for the lottery. At the
  // expense of not being that accurate with applying the selectivity an especially more memory
  // efficient implementations would be to partition the random numbers into two intervals and then
  // let one table choose random numbers from only one interval and the other only select with
  // selective probability from the same interval and from the other in the other cases.

  constexpr int block_size = 128;

  // Maximize exposed parallelism while minimizing storage for curand state
  int num_blocks_init_build_tbl{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_init_build_tbl, init_build_tbl<key_type, size_type>, block_size, 0));

  int num_blocks_init_probe_tbl{-1};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_init_probe_tbl, init_probe_tbl<key_type, size_type>, block_size, 0));

  int dev_id{-1};
  CUDF_CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  int const num_states =
    num_sms * std::max(num_blocks_init_build_tbl, num_blocks_init_probe_tbl) * block_size;
  rmm::device_uvector<curandState> devStates(num_states, cudf::get_default_stream());

  init_curand<<<(num_states - 1) / block_size + 1, block_size>>>(devStates.data(), num_states);

  CUDF_CHECK_CUDA(0);

  init_build_tbl<key_type, size_type><<<num_sms * num_blocks_init_build_tbl, block_size>>>(
    build_tbl, build_tbl_size, multiplicity, devStates.data(), num_states);

  CUDF_CHECK_CUDA(0);

  auto const rand_max = std::numeric_limits<key_type>::max();

  init_probe_tbl<key_type, size_type>
    <<<num_sms * num_blocks_init_build_tbl, block_size>>>(probe_tbl,
                                                          probe_tbl_size,
                                                          build_tbl_size,
                                                          rand_max,
                                                          selectivity,
                                                          multiplicity,
                                                          devStates.data(),
                                                          num_states);

  CUDF_CHECK_CUDA(0);
}
