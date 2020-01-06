/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


#ifndef __GENERATE_INPUT_TABLES_CUH
#define __GENERATE_INPUT_TABLES_CUH

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/distance.h>
#include <cassert>

#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <cudf/utilities/error.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>


__global__ static void init_curand(curandState * state, const int nstates)
{
    int ithread = threadIdx.x + blockIdx.x * blockDim.x;

    if (ithread < nstates) {
        curand_init(1234ULL, ithread, 0, state + ithread);
    }
}


template<typename key_type, typename size_type>
__global__ static void init_build_tbl(
    key_type* const build_tbl, const size_type build_tbl_size,
    const key_type rand_max,
    const bool uniq_build_tbl_keys,
    key_type* const lottery, const size_type lottery_size,
    curandState * state, const int num_states)
{
    static_assert(std::is_signed<key_type>::value, "key_type needs to be signed for lottery to work");

    const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const key_type stride = blockDim.x * gridDim.x;
    assert(start_idx < num_states);

    curandState localState = state[start_idx];

    for (size_type idx = start_idx; idx < build_tbl_size; idx += stride) {
        const double x = curand_uniform_double(&localState);

        if (uniq_build_tbl_keys) {
	        // If the build table keys need to be unique, go through lottery array from lottery_idx until finding a key
            // which has not been used (-1). Mark the key as been used by atomically setting the spot to -1.

            size_type lottery_idx = x * lottery_size;
            key_type lottery_val = -1;

            while (-1 == lottery_val) {
                lottery_val = lottery[lottery_idx];

                if (-1 != lottery_val) {
                    lottery_val = atomicCAS<key_type>(lottery + lottery_idx, lottery_val, -1);
                }

                lottery_idx = (lottery_idx + 1) % lottery_size;
            }

            build_tbl[idx] = lottery_val;
        } else {
            build_tbl[idx] = x * rand_max;
        }
    }

    state[start_idx] = localState;
}


template<typename key_type, typename size_type>
__global__ void init_probe_tbl(
                               key_type* const probe_tbl,
                               const size_type probe_tbl_size,
                               const key_type* const build_tbl,
                               const size_type build_tbl_size,
                               const key_type* const lottery,
                               const size_type lottery_size,
                               const double selectivity,
                               curandState * state,
                               const int num_states)
{
    const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    assert(start_idx < num_states);

    curandState localState = state[start_idx];

    for (size_type idx = start_idx; idx < probe_tbl_size; idx += stride) {
        key_type val;
        double x = curand_uniform_double(&localState);

        if (x <= selectivity) {
            // x <= selectivity means this key in the probe table should be present in the build table, so we pick a
            // key from build_tbl
            x = curand_uniform_double(&localState);
            size_type build_tbl_idx = x * build_tbl_size;

            if (build_tbl_idx >= build_tbl_size) {
                build_tbl_idx = build_tbl_size - 1;
            }

            val = build_tbl[build_tbl_idx];
        } else {
            // This key in the probe table should not be present in the build table, so we pick a key from lottery.
            x = curand_uniform_double(&localState);
            size_type lottery_idx = x * lottery_size;
            val = lottery[lottery_idx];
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
 * table with a propability of selectivity and a random number from
 * [0,rand_max] \setminus \{build_tbl\} otherwise.
 *
 * @param[out] build_tbl            The build table to generate. Usually the smaller table used to
 *                                  "build" the hash table in a hash based join implementation.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[out] probe_tbl            The probe table to generate. Usually the larger table used to
 *                                  probe into the hash table created from the build table.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[in] selectivity           propability with which an element of the probe table is
 *                                  present in the build table.
 * @param[in] rand_max              maximum random number to generate. I.e. random numbers are
 *                                  integers from [0,rand_max].
 * @param[in] uniq_build_tbl_keys   if each key in the build table should appear exactly once.
 */
template<typename key_type, typename size_type>
void generate_input_tables(
                           key_type* const build_tbl,
                           const size_type build_tbl_size,
                           key_type* const probe_tbl,
                           const size_type probe_tbl_size,
                           const double selectivity,
                           const key_type rand_max,
                           const bool uniq_build_tbl_keys)
{
    // With large values of rand_max the a lot of temporary storage is needed for the lottery. At the expense of not
    // being that accurate with applying the selectivity an especially more memory efficient implementations would be
    // to partition the random numbers into two intervals and then let one table choose random numbers from only one
    // interval and the other only select with selectivity propability from the same interval and from the other in the
    // other cases.

    static_assert(std::is_signed<key_type>::value, "key_type needs to be signed for lottery to work");

    const int block_size = 128;

    // Maximize exposed parallelism while minimizing storage for curand state
    int num_blocks_init_build_tbl {-1};
    CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_init_build_tbl, init_build_tbl<key_type, size_type>, block_size, 0
    ));

    int num_blocks_init_probe_tbl {-1};
    CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_init_probe_tbl, init_probe_tbl<key_type,size_type>, block_size, 0
    ));

    int dev_id {-1};
    CUDA_TRY(cudaGetDevice(&dev_id));

    int num_sms {-1};
    CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

    const int num_states = num_sms * std::max(num_blocks_init_build_tbl, num_blocks_init_probe_tbl) * block_size;
    curandState *devStates;
    RMM_TRY(RMM_ALLOC(&devStates, num_states * sizeof(curandState), 0));

    init_curand<<<(num_states - 1) / block_size + 1, block_size>>>(devStates, num_states);

    CHECK_CUDA(0);

    key_type* build_tbl_sorted;
    RMM_TRY(RMM_ALLOC(&build_tbl_sorted, build_tbl_size * sizeof(key_type), 0));

    size_type lottery_size = rand_max < std::numeric_limits<key_type>::max() - 1 ? rand_max + 1 : rand_max;
    key_type* lottery;

    RMM_TRY(RMM_ALLOC(&lottery, lottery_size * sizeof(key_type), 0));

    if (uniq_build_tbl_keys) {
        thrust::sequence(thrust::device, lottery, lottery + lottery_size, 0);
    }

    init_build_tbl<key_type, size_type><<<num_sms * num_blocks_init_build_tbl, block_size>>>(
        build_tbl, build_tbl_size, rand_max, uniq_build_tbl_keys,
        lottery, lottery_size, devStates, num_states
    );

    CHECK_CUDA(0);

    CUDA_TRY(cudaMemcpy(
        build_tbl_sorted, build_tbl, build_tbl_size * sizeof(key_type), cudaMemcpyDeviceToDevice
    ));

    thrust::sort(thrust::device, build_tbl_sorted, build_tbl_sorted + build_tbl_size);

    // Exclude keys used in build table from lottery
    thrust::counting_iterator<key_type> first_lottery_elem(0);
    thrust::counting_iterator<key_type> last_lottery_elem = first_lottery_elem + lottery_size;
    key_type *lottery_end = thrust::set_difference(
        thrust::device, first_lottery_elem, last_lottery_elem,
        build_tbl_sorted, build_tbl_sorted + build_tbl_size, lottery
    );

    lottery_size = thrust::distance(lottery, lottery_end);

    init_probe_tbl<key_type, size_type><<<num_sms * num_blocks_init_build_tbl, block_size>>>(
        probe_tbl, probe_tbl_size, build_tbl, build_tbl_size,
        lottery, lottery_size, selectivity, devStates, num_states
    );

    CHECK_CUDA(0);

    RMM_TRY(RMM_FREE(lottery, 0));
    RMM_TRY(RMM_FREE(build_tbl_sorted, 0));
    RMM_TRY(RMM_FREE(devStates, 0));
}


#endif  // __GENERATE_INPUT_TABLES_CUH
