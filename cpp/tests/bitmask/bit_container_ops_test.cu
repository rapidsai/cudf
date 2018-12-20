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
#include "bitmask/bit_container.cuh"

#include <chrono>

using bit_size_t = bit_container::bit_size_t;

struct BitContainerTest : public GdfTest {};

__global__ void count_bits(bit_size_t *counter, BitContainer bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
    
  bit_size_t local_counter = 0;

  for (int i = index ; i < bits.NumWords() ; i += stride) {
    local_counter += __popc(bits.GetValid()[i]);
  }

  atomicAdd(counter, local_counter);
}

//
//  Testing function, will set a bit in a container.  This assumes <1,1>
//  for simplicity - all of the tests are small.
//
__global__ void set_bit(bit_size_t bit, BitContainer bits) {
  bits.SetBit(bit);
}

__host__ bit_container::bit_size_t count_bits(const BitContainer &bit_container, int a = 1, int b = 1) {
  bit_container::bit_size_t *count;
  cudaMalloc((void **) &count, sizeof(bit_container::bit_size_t));
  cudaMemset(count, 0, sizeof(bit_container::bit_size_t));
  
  count_bits<<<a,b>>>(count, bit_container);

  bit_container::bit_size_t local_count;
  cudaMemcpy(&local_count, count, sizeof(bit_container::bit_size_t), cudaMemcpyDeviceToHost);
  cudaFree(count);

  return local_count;
}


TEST_F(BitContainerTest, NoValids)
{
  const int num_rows = 100;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(0U, local_count);
}

TEST_F(BitContainerTest, AllValids)
{
  const int num_rows = 100;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 1);

  BitContainer bit_container(bits, num_rows);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(100U, local_count);
}

TEST_F(BitContainerTest, FirstRowValid)
{
  const int num_rows = 4;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_container);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  EXPECT_EQ(1U, local_count);

  bit_container_t temp = 0;
  bit_container.GetWord(&temp, bit_container.GetValid());

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(temp, 0x1U);
}

TEST_F(BitContainerTest, EveryOtherBit)
{
  const int num_rows = 8;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_container);
  set_bit<<<1,1>>>(2, bit_container);
  set_bit<<<1,1>>>(4, bit_container);
  set_bit<<<1,1>>>(6, bit_container);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  EXPECT_EQ(4U, local_count);

  bit_container_t temp = 0;
  bit_container.GetWord(&temp, bit_container.GetValid());

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(temp, 0x55U);
}

TEST_F(BitContainerTest, OtherEveryOtherBit)
{
  const int num_rows = 8;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  set_bit<<<1,1>>>(1, bit_container);
  set_bit<<<1,1>>>(3, bit_container);
  set_bit<<<1,1>>>(5, bit_container);
  set_bit<<<1,1>>>(7, bit_container);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  EXPECT_EQ(4U, local_count);

  bit_container_t temp = 0;
  bit_container.GetWord(&temp, bit_container.GetValid());

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(temp, 0xAAU);
}

TEST_F(BitContainerTest, 15rows)
{
  const int num_rows = 15;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_container);
  set_bit<<<1,1>>>(8, bit_container);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  EXPECT_EQ(2U, local_count);

  BitContainer::DestroyBitContainer(bits);
}

TEST_F(BitContainerTest, 5rows)
{
  const int num_rows = 5;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  set_bit<<<1,1>>>(0, bit_container);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(1U, local_count);
}

TEST_F(BitContainerTest, 10ValidRows)
{
  const int num_rows = 10;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 1);

  BitContainer bit_container(bits, num_rows);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(10U, local_count);
}

TEST_F(BitContainerTest, MultipleOfEight)
{
  const int num_rows = 1024;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  for (int i = 0 ; i < num_rows ; i += 8) {
    set_bit<<<1,1>>>(i, bit_container);
  }
  
  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(128U, local_count);
}

TEST_F(BitContainerTest, NotMultipleOfEight)
{
  const int num_rows = 1023;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  for (int i = 7 ; i < num_rows ; i += 8) {
    set_bit<<<1,1>>>(i, bit_container);
  }
  
  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(127U, local_count);
}

TEST_F(BitContainerTest, TenThousandRows)
{
  const int num_rows = 10000;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 1);

  BitContainer bit_container(bits, num_rows);

  bit_container::bit_size_t local_count = count_bits(bit_container);

  BitContainer::DestroyBitContainer(bits);

  EXPECT_EQ(10000U, local_count);
}

TEST_F(BitContainerTest, PerformanceTest)
{
  const int num_rows = 100000000;

  bit_container_t *bits = BitContainer::CreateBitContainer(num_rows, 0);

  BitContainer bit_container(bits, num_rows);

  int num_words = bit_container::num_words(num_rows);
  int block_size = 256;
  int grid_size = (num_words + block_size - 1)/block_size;

  uint32_t *local_valid = (uint32_t *) malloc(num_words * sizeof(uint32_t));
  for (int i = 0 ; i < num_words ; ++i) {
    local_valid[i] = 0x55555555U;
  }

  BitContainer::CopyBitContainer(bit_container.GetValid(), local_valid, num_rows, cudaMemcpyHostToDevice);

  auto start = std::chrono::system_clock::now();
  cudaProfilerStart();
  for(int i = 0; i < 1000; ++i) {
    count_bits(bit_container, grid_size, block_size);
  }
  cudaProfilerStop();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;

  BitContainer::DestroyBitContainer(bits);
  free(local_valid);
}
