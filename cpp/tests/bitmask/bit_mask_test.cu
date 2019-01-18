/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "cuda_profiler_api.h"

#include "tests/utilities/cudf_test_utils.cuh"
#include "tests/utilities/cudf_test_fixtures.h"
#include "bitmask/bit_mask.cuh"

#include <chrono>

struct BitMaskTest : public GdfTest {};

//
//  Kernel to count bits set in the bit mask
//
__global__ void count_bits_g(int *counter, BitMask bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
    
  int local_counter = 0;
  int i;

  for (i = index ; i < (bits.num_elements() - 1) ; i += stride) {
    local_counter += __popc(bits.get_element_device(i));
  }

  if (i == (bits.num_elements() - 1)) {
    //
    //  Special case... last word is only partial
    //
    int bits_used = bits.length() % bit_mask::detail::BITS_PER_ELEMENT;
    if (bits_used == 0) {
      //
      //  The whole word is used
      //
      local_counter += __popc(bits.get_element_device(i));
    } else {
      local_counter += __popc(bits.get_element_device(i) & ((bit_mask_t{1} << bits_used) - 1));
    }
  }

  atomicAdd(counter, local_counter);
}

//
//  Testing function, will set a bit in a container.  This assumes <1,1>
//  for simplicity - all of the tests are small.
//
__global__ void set_bit(gdf_size_type bit, BitMask bits) {
  bits.set_bit_unsafe(bit);
}

//
//  Kernel to do safe bit set/clear
//
__global__ void test_safe_set_clear_g(BitMask bits) {
  int index = threadIdx.x;

  if ((index % 2) == 0) {
    for (int i = index ; i < bits.length() ; i += bit_mask::detail::BITS_PER_ELEMENT) {
      bits.set_bit(i);
    }
  }

  for (int i = index ; i < bits.length() ; i += bit_mask::detail::BITS_PER_ELEMENT) {
    bits.clear_bit(i);
  }

  if ((index % 2) == 0) {
    for (int i = index ; i < bits.length() ; i += bit_mask::detail::BITS_PER_ELEMENT) {
      bits.set_bit(i);
    }
  }
}


__host__ gdf_error count_bits(gdf_size_type *count, const BitMask &bit_mask, int a = 1, int b = 1) {
  int *count_d;
  CUDA_TRY(cudaMalloc(&count_d, sizeof(int)));
  CUDA_TRY(cudaMemset(count_d, 0, sizeof(int)));
  
  count_bits_g<<<a,b>>>(count_d, bit_mask);

  CUDA_TRY(cudaMemcpy(count, count_d, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaFree(count_d));

  return GDF_SUCCESS;
}


TEST_F(BitMaskTest, NoValids)
{
  const int num_rows = 100;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{0}, local_count);
}

TEST_F(BitMaskTest, AllValids)
{
  const int num_rows = 100;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 1));

  BitMask bit_mask(bits, num_rows);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{100}, local_count);
}

TEST_F(BitMaskTest, FirstRowValid)
{
  const int num_rows = 4;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(gdf_size_type{1}, local_count);

  bit_mask_t temp = 0;
  bit_mask.get_element_host(0, temp);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(temp, bit_mask_t{0x1});
}

TEST_F(BitMaskTest, EveryOtherBit)
{
  const int num_rows = 8;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_mask);
  set_bit<<<1,1>>>(2, bit_mask);
  set_bit<<<1,1>>>(4, bit_mask);
  set_bit<<<1,1>>>(6, bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(gdf_size_type{4}, local_count);

  bit_mask_t temp = 0;
  bit_mask.get_element_host(0, temp);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(temp, bit_mask_t{0x55});
}

TEST_F(BitMaskTest, OtherEveryOtherBit)
{
  const int num_rows = 8;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  set_bit<<<1,1>>>(1, bit_mask);
  set_bit<<<1,1>>>(3, bit_mask);
  set_bit<<<1,1>>>(5, bit_mask);
  set_bit<<<1,1>>>(7, bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(gdf_size_type{4}, local_count);

  bit_mask_t temp = 0;
  bit_mask.get_element_host(0, temp);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(temp, bit_mask_t{0xAA});
}

TEST_F(BitMaskTest, 15rows)
{
  const int num_rows = 15;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_mask);
  set_bit<<<1,1>>>(8, bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(gdf_size_type{2}, local_count);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));
}

TEST_F(BitMaskTest, 5rows)
{
  const int num_rows = 5;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{1}, local_count);
}

TEST_F(BitMaskTest, 10ValidRows)
{
  const int num_rows = 10;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 1));

  BitMask bit_mask(bits, num_rows);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{10}, local_count);
}

TEST_F(BitMaskTest, MultipleOfEight)
{
  const int num_rows = 1024;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  for (int i = 0 ; i < num_rows ; i += 8) {
    set_bit<<<1,1>>>(i, bit_mask);
  }
  
  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{128}, local_count);
}

TEST_F(BitMaskTest, NotMultipleOfEight)
{
  const int num_rows = 1023;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  for (int i = 7 ; i < num_rows ; i += 8) {
    set_bit<<<1,1>>>(i, bit_mask);
  }
  
  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{127}, local_count);
}

TEST_F(BitMaskTest, TenThousandRows)
{
  const int num_rows = 10000;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 1));

  BitMask bit_mask(bits, num_rows);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));

  EXPECT_EQ(gdf_size_type{10000}, local_count);
}

TEST_F(BitMaskTest, PerformanceTest)
{
  const int num_rows = 100000000;

  bit_mask_t *bits = nullptr;
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  int num_elements = bit_mask::num_elements(num_rows);
  int block_size = 256;
  int grid_size = (num_elements + block_size - 1)/block_size;

  uint32_t *local_valid = (uint32_t *) malloc(num_elements * sizeof(uint32_t));
  for (int i = 0 ; i < num_elements ; ++i) {
    local_valid[i] = 0x55555555U;
  }

  EXPECT_EQ(GDF_SUCCESS, bit_mask::copy_bit_mask(bit_mask.get_valid(), local_valid, num_rows, cudaMemcpyHostToDevice));

  auto start = std::chrono::system_clock::now();
  cudaProfilerStart();
  for(int i = 0; i < 1000; ++i) {
    gdf_size_type local_count = 0;
    count_bits(&local_count, bit_mask, grid_size, block_size);
  }
  cudaProfilerStop();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));
  free(local_valid);
}

TEST_F(BitMaskTest, CudaThreadingTest)
{
  const int num_rows = 100000;
  bit_mask_t *bits = nullptr;
 
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 0));

  BitMask bit_mask(bits, num_rows);

  test_safe_set_clear_g<<<1,bit_mask::detail::BITS_PER_ELEMENT>>>(bit_mask);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ((gdf_size_type) (num_rows/2), local_count);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));
}

TEST_F(BitMaskTest, PaddingTest)
{
  //
  //  Set the number of rows to 32, we'll try padding to 
  //  256 bytes.
  //
  const int num_rows = 32;
  const int padding_bytes = 256;
  bit_mask_t *bits = nullptr;
 
  EXPECT_EQ(GDF_SUCCESS, bit_mask::create_bit_mask(&bits, num_rows, 1, padding_bytes));

  BitMask bit_mask(bits, num_rows);

  gdf_size_type local_count = 0;
  EXPECT_EQ(GDF_SUCCESS, count_bits(&local_count, bit_mask));

  EXPECT_EQ((gdf_size_type) num_rows, local_count);

  //
  //  To test this, we should be able to access the last element
  //
  int last_element = (padding_bytes / sizeof(bit_mask_t)) - 1;
  
  bit_mask_t temp = bit_mask_t{0};
  bit_mask.get_element_host(last_element, temp);
  EXPECT_EQ(~bit_mask_t{0}, temp);

  EXPECT_EQ(GDF_SUCCESS, bit_mask::destroy_bit_mask(bits));
}
