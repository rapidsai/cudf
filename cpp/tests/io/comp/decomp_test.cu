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

#include <io/comp/gpuinflate.h>
#include <tests/utilities/base_fixture.hpp>

#include <vector>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

/**
 * @brief Base test fixture for decompression
 *
 * Calls into Decompressor fixture to dispatch actual decompression work,
 * whose interface and setup is different for each codec.
 **/
template <typename Decompressor>
struct DecompressTest : public cudf::test::BaseFixture {
  void SetUp() override
  {
    ASSERT_CUDA_SUCCEEDED(cudaMallocHost((void**)&inf_args, sizeof(cudf::io::gpu_inflate_input_s)));
    ASSERT_CUDA_SUCCEEDED(
      cudaMallocHost((void**)&inf_stat, sizeof(cudf::io::gpu_inflate_status_s)));

    d_inf_args.resize(1);
    d_inf_stat.resize(1);
  }

  void TearDown() override
  {
    ASSERT_CUDA_SUCCEEDED(cudaFreeHost(inf_stat));
    ASSERT_CUDA_SUCCEEDED(cudaFreeHost(inf_args));
  }

  std::vector<uint8_t> vector_from_string(const char* str) const
  {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(str),
                                reinterpret_cast<const uint8_t*>(str) + strlen(str));
  }

  void Decompress(std::vector<uint8_t>* decompressed,
                  const uint8_t* compressed,
                  size_t compressed_size)
  {
    rmm::device_buffer src(compressed, compressed_size);
    rmm::device_buffer dst(decompressed->size());

    inf_args->srcDevice = static_cast<const uint8_t*>(src.data());
    inf_args->dstDevice = static_cast<uint8_t*>(dst.data());
    inf_args->srcSize   = src.size();
    inf_args->dstSize   = dst.size();
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_inf_args.data().get(),
                                          inf_args,
                                          sizeof(cudf::io::gpu_inflate_input_s),
                                          cudaMemcpyHostToDevice,
                                          0));
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_inf_stat.data().get(),
                                          inf_stat,
                                          sizeof(cudf::io::gpu_inflate_status_s),
                                          cudaMemcpyHostToDevice,
                                          0));
    ASSERT_CUDA_SUCCEEDED(static_cast<Decompressor*>(this)->dispatch());
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(inf_stat,
                                          d_inf_stat.data().get(),
                                          sizeof(cudf::io::gpu_inflate_status_s),
                                          cudaMemcpyDeviceToHost,
                                          0));
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(
      decompressed->data(), inf_args->dstDevice, inf_args->dstSize, cudaMemcpyDeviceToHost, 0));
    ASSERT_CUDA_SUCCEEDED(cudaStreamSynchronize(0));
  }

  cudf::io::gpu_inflate_input_s* inf_args  = nullptr;
  cudf::io::gpu_inflate_status_s* inf_stat = nullptr;
  rmm::device_vector<cudf::io::gpu_inflate_input_s> d_inf_args;
  rmm::device_vector<cudf::io::gpu_inflate_status_s> d_inf_stat;
};

/**
 * @brief Derived fixture for GZIP decompression
 **/
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  cudaError_t dispatch()
  {
    return cudf::io::gpuinflate(d_inf_args.data().get(), d_inf_stat.data().get(), 1, 1);
  }
};

/**
 * @brief Derived fixture for Snappy decompression
 **/
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  cudaError_t dispatch()
  {
    return cudf::io::gpu_unsnap(d_inf_args.data().get(), d_inf_stat.data().get(), 1);
  }
};

/**
 * @brief Derived fixture for Brotli decompression
 **/
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  cudaError_t dispatch()
  {
    rmm::device_buffer d_scratch(cudf::io::get_gpu_debrotli_scratch_size(1));

    return cudf::io::gpu_debrotli(
      d_inf_args.data().get(), d_inf_stat.data().get(), d_scratch.data(), d_scratch.size(), 1);
  }
};

TEST_F(GzipDecompressTest, HelloWorld)
{
  constexpr char uncompressed[]  = "hello world";
  constexpr uint8_t compressed[] = {
    0x1f, 0x8b, 0x8,  0x0,  0x9,  0x63, 0x99, 0x5c, 0x2,  0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57,
    0x28, 0xcf, 0x2f, 0xca, 0x49, 0x1,  0x0,  0x85, 0x11, 0x4a, 0xd,  0xb,  0x0,  0x0,  0x0};

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  std::vector<uint8_t> output(input.size());
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_EQ(output, input);
}

TEST_F(SnappyDecompressTest, HelloWorld)
{
  constexpr char uncompressed[]  = "hello world";
  constexpr uint8_t compressed[] = {
    0xb, 0x28, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64};

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  std::vector<uint8_t> output(input.size());
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_EQ(output, input);
}

TEST_F(SnappyDecompressTest, ShortLiteralAfterLongCopyAtStartup)
{
  constexpr char uncompressed[]  = "Aaaaaaaaaaaah!";
  constexpr uint8_t compressed[] = {14, 0x0, 'A', 0x0, 'a', (10 - 4) * 4 + 1, 1, 0x4, 'h', '!'};

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  std::vector<uint8_t> output(input.size());
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_EQ(output, input);
}

TEST_F(BrotliDecompressTest, HelloWorld)
{
  constexpr char uncompressed[]  = "hello world";
  constexpr uint8_t compressed[] = {
    0xb, 0x5, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x3};

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  std::vector<uint8_t> output(input.size());
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_EQ(output, input);
}

CUDF_TEST_PROGRAM_MAIN()
