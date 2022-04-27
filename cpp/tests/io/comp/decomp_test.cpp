/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <io/utilities/hostdevice_vector.hpp>

#include <cudf_test/base_fixture.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <vector>

using cudf::device_span;

/**
 * @brief Base test fixture for decompression
 *
 * Calls into Decompressor fixture to dispatch actual decompression work,
 * whose interface and setup is different for each codec.
 */
template <typename Decompressor>
struct DecompressTest : public cudf::test::BaseFixture {
  std::vector<uint8_t> vector_from_string(const char* str) const
  {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(str),
                                reinterpret_cast<const uint8_t*>(str) + strlen(str));
  }

  void Decompress(std::vector<uint8_t>* decompressed,
                  const uint8_t* compressed,
                  size_t compressed_size)
  {
    auto stream = rmm::cuda_stream_default;
    rmm::device_buffer src{compressed, compressed_size, stream};
    rmm::device_uvector<uint8_t> dst{decompressed->size(), stream};

    hostdevice_vector<device_span<uint8_t const>> inf_in(1, stream);
    inf_in[0] = {static_cast<uint8_t const*>(src.data()), src.size()};
    inf_in.host_to_device(stream);

    hostdevice_vector<device_span<uint8_t>> inf_out(1, stream);
    inf_out[0] = dst;
    inf_out.host_to_device(stream);

    hostdevice_vector<cudf::io::decompress_status> inf_stat(1, stream);
    inf_stat[0] = {};
    inf_stat.host_to_device(stream);

    static_cast<Decompressor*>(this)->dispatch(inf_in, inf_out, inf_stat);
    cudaMemcpyAsync(
      decompressed->data(), dst.data(), dst.size(), cudaMemcpyDeviceToHost, stream.value());
    inf_stat.device_to_host(stream, true);
    ASSERT_EQ(inf_stat[0].status, 0);
  }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::decompress_status> d_inf_stat)
  {
    cudf::io::gpuinflate(d_inf_in,
                         d_inf_out,
                         d_inf_stat,
                         cudf::io::gzip_header_included::YES,
                         rmm::cuda_stream_default);
  }
};

/**
 * @brief Derived fixture for Snappy decompression
 */
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::decompress_status> d_inf_stat)
  {
    cudf::io::gpu_unsnap(d_inf_in, d_inf_out, d_inf_stat, rmm::cuda_stream_default);
  }
};

/**
 * @brief Derived fixture for Brotli decompression
 */
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::decompress_status> d_inf_stat)
  {
    rmm::device_buffer d_scratch{cudf::io::get_gpu_debrotli_scratch_size(1),
                                 rmm::cuda_stream_default};

    cudf::io::gpu_debrotli(d_inf_in,
                           d_inf_out,
                           d_inf_stat,
                           d_scratch.data(),
                           d_scratch.size(),
                           rmm::cuda_stream_default);
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
