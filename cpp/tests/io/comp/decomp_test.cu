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
#include <tests/utilities/cudf_test_fixtures.h>

#include <vector>

/**
 * @brief Base test fixture for decompression
 *
 * Calls into Decompressor fixture to dispatch actual decompression work,
 * whose interface and setup is different for each codec.
 **/
template <typename Decompressor>
struct DecompressTest : public GdfTest {
  void SetUp() override {
    ASSERT_CUDA_SUCCEEDED(
        cudaMallocHost((void**)&inf_args, sizeof(gpu_inflate_input_s)));
    ASSERT_CUDA_SUCCEEDED(
        cudaMallocHost((void**)&inf_stat, sizeof(gpu_inflate_status_s)));
    ASSERT_RMM_SUCCEEDED(
        RMM_ALLOC(&d_inf_args, sizeof(gpu_inflate_input_s), 0));
    ASSERT_RMM_SUCCEEDED(
        RMM_ALLOC(&d_inf_stat, sizeof(gpu_inflate_status_s), 0));
  }

  void TearDown() override {
    RMM_FREE(d_inf_stat, 0);
    RMM_FREE(d_inf_args, 0);
    cudaFreeHost(inf_stat);
    cudaFreeHost(inf_args);
  }

  void Decompress(std::vector<uint8_t>* decompressed, const uint8_t* compressed,
                  size_t compressed_size) {
    inf_args->srcSize = compressed_size;
    inf_args->dstSize = decompressed->size();
    ASSERT_RMM_SUCCEEDED(RMM_ALLOC(&inf_args->srcDevice, inf_args->srcSize, 0));
    ASSERT_RMM_SUCCEEDED(RMM_ALLOC(&inf_args->dstDevice, inf_args->dstSize, 0));
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(inf_args->srcDevice, compressed,
                                          inf_args->srcSize,
                                          cudaMemcpyHostToDevice, 0));

    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_inf_args, inf_args,
                                          sizeof(gpu_inflate_input_s),
                                          cudaMemcpyHostToDevice, 0));
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(d_inf_stat, inf_stat,
                                          sizeof(gpu_inflate_status_s),
                                          cudaMemcpyHostToDevice, 0));
    ASSERT_CUDA_SUCCEEDED(static_cast<Decompressor*>(this)->dispatch());
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(inf_stat, d_inf_stat,
                                          sizeof(gpu_inflate_status_s),
                                          cudaMemcpyDeviceToHost, 0));
    ASSERT_CUDA_SUCCEEDED(cudaMemcpyAsync(decompressed->data(),
                                          inf_args->dstDevice, inf_args->dstSize,
                                          cudaMemcpyDeviceToHost, 0));
    ASSERT_CUDA_SUCCEEDED(cudaStreamSynchronize(0));

    ASSERT_RMM_SUCCEEDED(RMM_FREE(inf_args->dstDevice, 0));
    ASSERT_RMM_SUCCEEDED(RMM_FREE(inf_args->srcDevice, 0));
  }

  gpu_inflate_input_s* inf_args = nullptr;
  gpu_inflate_status_s* inf_stat = nullptr;
  gpu_inflate_input_s* d_inf_args = nullptr;
  gpu_inflate_status_s* d_inf_stat = nullptr;
};

/**
 * @brief Derived fixture for GZIP decompression
 **/
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  cudaError_t dispatch() { return gpuinflate(d_inf_args, d_inf_stat, 1, 1); }
};

/**
 * @brief Derived fixture for Snappy decompression
 **/
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  cudaError_t dispatch() { return gpu_unsnap(d_inf_args, d_inf_stat, 1); }
};

/**
 * @brief Derived fixture for Brotli decompression
 **/
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  cudaError_t dispatch() {
    uint8_t* d_scratch = nullptr;
    size_t scratch_size = get_gpu_debrotli_scratch_size(1);

    RMM_ALLOC(&d_scratch, scratch_size, 0);
    auto ret = gpu_debrotli(d_inf_args, d_inf_stat, d_scratch, scratch_size, 1);
    RMM_FREE(d_scratch, 0);
    return ret;
  }
};

TEST_F(GzipDecompressTest, HelloWorld) {
  constexpr char uncompressed[] = "hello world";
  constexpr uint8_t compressed[] = {
      0x1f, 0x8b, 0x8,  0x0,  0x9,  0x63, 0x99, 0x5c, 0x2,  0xff, 0xcb,
      0x48, 0xcd, 0xc9, 0xc9, 0x57, 0x28, 0xcf, 0x2f, 0xca, 0x49, 0x1,
      0x0,  0x85, 0x11, 0x4a, 0xd,  0xb,  0x0,  0x0,  0x0};

  std::vector<uint8_t> output(sizeof(uncompressed));
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), uncompressed);
}

TEST_F(SnappyDecompressTest, HelloWorld) {
  constexpr char uncompressed[] = "hello world";
  constexpr uint8_t compressed[] = {0xb,  0x28, 0x68, 0x65, 0x6c, 0x6c, 0x6f,
                                    0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64};

  std::vector<uint8_t> output(sizeof(uncompressed));
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), uncompressed);
}

TEST_F(BrotliDecompressTest, HelloWorld) {
  constexpr char uncompressed[] = "hello world";
  constexpr uint8_t compressed[] = {0xb,  0x5,  0x80, 0x68, 0x65,
                                    0x6c, 0x6c, 0x6f, 0x20, 0x77,
                                    0x6f, 0x72, 0x6c, 0x64, 0x3};

  std::vector<uint8_t> output(sizeof(uncompressed));
  Decompress(&output, compressed, sizeof(compressed));
  EXPECT_STREQ(reinterpret_cast<char*>(output.data()), uncompressed);
}
