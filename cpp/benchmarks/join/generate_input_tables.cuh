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
#include <utilities/error_utils.hpp>
#include <utilities/device_atomics.cuh>


template <typename col_type>
gdf_dtype gdf_dtype_from_col_type()
{
    if(std::is_same<col_type,int8_t>::value) return GDF_INT8;
    else if(std::is_same<col_type,uint8_t>::value) return GDF_INT8;
    else if(std::is_same<col_type,int16_t>::value) return GDF_INT16;
    else if(std::is_same<col_type,uint16_t>::value) return GDF_INT16;
    else if(std::is_same<col_type,int32_t>::value) return GDF_INT32;
    else if(std::is_same<col_type,uint32_t>::value) return GDF_INT32;
    else if(std::is_same<col_type,int64_t>::value) return GDF_INT64;
    else if(std::is_same<col_type,uint64_t>::value) return GDF_INT64;
    else if(std::is_same<col_type,float>::value) return GDF_FLOAT32;
    else if(std::is_same<col_type,double>::value) return GDF_FLOAT64;
    else return GDF_invalid;
}


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
    CUDA_TRY(cudaMalloc(&devStates, num_states * sizeof(curandState)));

    init_curand<<<(num_states - 1) / block_size + 1, block_size>>>(devStates, num_states);

    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    key_type* build_tbl_sorted;
    CUDA_TRY(cudaMalloc(&build_tbl_sorted, build_tbl_size * sizeof(key_type)));

    size_type lottery_size = rand_max < std::numeric_limits<key_type>::max() - 1 ? rand_max + 1 : rand_max;
    key_type* lottery;
    bool lottery_in_device_memory = true;

    size_t free_gpu_mem {0};
    size_t total_gpu_mem {0};

    CUDA_TRY(cudaMemGetInfo(&free_gpu_mem, &total_gpu_mem));

    if (free_gpu_mem > lottery_size * sizeof(key_type)) {
        CUDA_TRY(cudaMalloc(&lottery, lottery_size * sizeof(key_type)));
    } else {
        CUDA_TRY(cudaMallocHost(&lottery, lottery_size * sizeof(key_type)));
        lottery_in_device_memory = false;
    }

    if (uniq_build_tbl_keys) {
        thrust::sequence(thrust::device, lottery, lottery + lottery_size, 0);
    }

    init_build_tbl<key_type, size_type><<<num_sms * num_blocks_init_build_tbl, block_size>>>(
        build_tbl, build_tbl_size, rand_max, uniq_build_tbl_keys,
        lottery, lottery_size, devStates, num_states
    );

    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

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

    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    if (lottery_in_device_memory) {
        CUDA_TRY(cudaFree(lottery));
    } else {
        CUDA_TRY(cudaFreeHost(lottery));
    }

    CUDA_TRY(cudaFree(build_tbl_sorted));
    CUDA_TRY(cudaFree(devStates));
}


template<typename key_type, typename size_type>
__global__ void linear_sequence(key_type* tbl, const size_type size)
{
  for (size_type i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x)
    tbl[i] = i;
}


/**
 * Generate a build table and a probe table for testing join performance.
 *
 * Both the build table and the probe table have two columns. The first column is the key column,
 * with datatype KEY_T. The second column is the payload column, with datatype PAYLOAD_T. Both the
 * arguments build_table and probe_table do not need to be preallocated. It is the caller's
 * responsibility to free memory of build_table and probe_table allocated by this function.
 *
 * @param[out] build_table         The build table to generate.
 * @param[in] build_table_size     The number of rows in the build table.
 * @param[out] probe_table         The probe table to generate.
 * @param[in] probe_table_size     The number of rows in the probe table.
 * @param[in] selectivity          Propability with which an element of the probe table is present in
 *                                 the build table.
 * @param[in] rand_max             Maximum random number to generate, i.e., random numbers are
 *                                 integers from [0, rand_max].
 * @param[in] uniq_build_tbl_keys  If each key in the build table should appear exactly once.
 */
template<typename KEY_T, typename PAYLOAD_T>
void generate_build_probe_tables(std::vector<gdf_column *> &build_table,
                                 gdf_size_type build_table_size,
                                 std::vector<gdf_column *> &probe_table,
                                 gdf_size_type probe_table_size,
                                 const double selectivity,
                                 const KEY_T rand_max,
                                 const bool uniq_build_tbl_keys)
{
    // Allocate device memory for generating data

    KEY_T *build_key_data {nullptr};
    PAYLOAD_T *build_payload_data {nullptr};
    KEY_T *probe_key_data {nullptr};
    PAYLOAD_T *probe_payload_data {nullptr};

    RMM_TRY(RMM_ALLOC(&build_key_data, build_table_size * sizeof(KEY_T), 0));

    RMM_TRY(RMM_ALLOC(&build_payload_data, build_table_size * sizeof(PAYLOAD_T), 0));

    RMM_TRY(RMM_ALLOC(&probe_key_data, probe_table_size * sizeof(KEY_T), 0));

    RMM_TRY(RMM_ALLOC(&probe_payload_data, probe_table_size * sizeof(PAYLOAD_T), 0));

    // Generate build and probe table data

    generate_input_tables<KEY_T, gdf_size_type>(
        build_key_data, build_table_size, probe_key_data, probe_table_size,
        selectivity, rand_max, uniq_build_tbl_keys
    );

    linear_sequence<PAYLOAD_T, gdf_size_type><<<(build_table_size+127)/128,128>>>(
        build_payload_data, build_table_size
    );

    linear_sequence<PAYLOAD_T, gdf_size_type><<<(probe_table_size+127)/128,128>>>(
        probe_payload_data, probe_table_size
    );

    CUDA_TRY(cudaGetLastError());
    CUDA_TRY(cudaDeviceSynchronize());

    // Generate build and probe table from data

    gdf_dtype gdf_key_t = gdf_dtype_from_col_type<KEY_T>();
    gdf_dtype gdf_payload_t = gdf_dtype_from_col_type<PAYLOAD_T>();

    build_table.resize(2, nullptr);

    for (auto & column_ptr : build_table) {
        column_ptr = new gdf_column;
    }

    CUDF_TRY(gdf_column_view(build_table[0], build_key_data, nullptr, build_table_size, gdf_key_t));

    CUDF_TRY(gdf_column_view(build_table[1], build_payload_data, nullptr, build_table_size, gdf_payload_t));

    probe_table.resize(2, nullptr);

    for (auto & column_ptr : probe_table) {
        column_ptr = new gdf_column;
    }

    CUDF_TRY(gdf_column_view(probe_table[0], probe_key_data, nullptr, probe_table_size, gdf_key_t));

    CUDF_TRY(gdf_column_view(probe_table[1], probe_payload_data, nullptr, probe_table_size, gdf_payload_t));
}


/**
 * Free the table as well as the device buffer it contains.
 *
 * @param[in] table    The table to be freed.
 */
void free_table(std::vector<gdf_column *> & table)
{
    for (auto & column_ptr : table) {
        CUDF_TRY(gdf_column_free(column_ptr));
        delete column_ptr;
    }
}


#endif  // __GENERATE_INPUT_TABLES_CUH
