/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <io/comp/gpuinflate.hpp>
#include <io/utilities/hostdevice_vector.hpp>
#include <src/io/comp/nvcomp_adapter.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/base_fixture.hpp>

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
  std::vector<uint8_t> vector_from_string(char const* str) const
  {
    return std::vector<uint8_t>(reinterpret_cast<uint8_t const*>(str),
                                reinterpret_cast<uint8_t const*>(str) + strlen(str));
  }

  void Decompress(std::vector<uint8_t>* decompressed,
                  uint8_t const* compressed,
                  size_t compressed_size)
  {
    auto stream = cudf::get_default_stream();
    rmm::device_buffer src{compressed, compressed_size, stream};
    rmm::device_uvector<uint8_t> dst{decompressed->size(), stream};

    cudf::detail::hostdevice_vector<device_span<uint8_t const>> inf_in(1, stream);
    inf_in[0] = {static_cast<uint8_t const*>(src.data()), src.size()};
    inf_in.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<device_span<uint8_t>> inf_out(1, stream);
    inf_out[0] = dst;
    inf_out.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<cudf::io::compression_result> inf_stat(1, stream);
    inf_stat[0] = {};
    inf_stat.host_to_device_async(stream);

    static_cast<Decompressor*>(this)->dispatch(inf_in, inf_out, inf_stat);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      decompressed->data(), dst.data(), dst.size(), cudaMemcpyDefault, stream.value()));
    inf_stat.device_to_host_sync(stream);
    ASSERT_EQ(inf_stat[0].status, cudf::io::compression_status::SUCCESS);
  }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::compression_result> d_inf_stat)
  {
    cudf::io::gpuinflate(d_inf_in,
                         d_inf_out,
                         d_inf_stat,
                         cudf::io::gzip_header_included::YES,
                         cudf::get_default_stream());
  }
};

/**
 * @brief Derived fixture for Snappy decompression
 */
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::compression_result> d_inf_stat)
  {
    cudf::io::gpu_unsnap(d_inf_in, d_inf_out, d_inf_stat, cudf::get_default_stream());
  }
};

/**
 * @brief Derived fixture for Brotli decompression
 */
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  void dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                device_span<device_span<uint8_t>> d_inf_out,
                device_span<cudf::io::compression_result> d_inf_stat)
  {
    rmm::device_buffer d_scratch{cudf::io::get_gpu_debrotli_scratch_size(1),
                                 cudf::get_default_stream()};

    cudf::io::gpu_debrotli(d_inf_in,
                           d_inf_out,
                           d_inf_stat,
                           d_scratch.data(),
                           d_scratch.size(),
                           cudf::get_default_stream());
  }
};

struct NvcompConfigTest : public cudf::test::BaseFixture {};

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

TEST_F(NvcompConfigTest, Compression)
{
  using cudf::io::nvcomp::compression_type;
  auto const& comp_disabled = cudf::io::nvcomp::is_compression_disabled;

  EXPECT_FALSE(comp_disabled(compression_type::DEFLATE, {2, 5, 0, true, true, 0}));
  // version 2.5 required
  EXPECT_TRUE(comp_disabled(compression_type::DEFLATE, {2, 4, 0, true, true, 0}));
  // all integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::DEFLATE, {2, 5, 0, false, true, 0}));

  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {2, 4, 0, true, true, 0}));
  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {2, 4, 0, false, true, 0}));
  // 2.4 version required
  EXPECT_TRUE(comp_disabled(compression_type::ZSTD, {2, 3, 1, false, true, 0}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::ZSTD, {2, 4, 0, false, false, 0}));

  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {2, 5, 0, true, true, 0}));
  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {2, 4, 0, false, true, 0}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::SNAPPY, {2, 3, 0, false, false, 0}));
}

TEST_F(NvcompConfigTest, Decompression)
{
  using cudf::io::nvcomp::compression_type;
  auto const& decomp_disabled = cudf::io::nvcomp::is_decompression_disabled;

  EXPECT_FALSE(decomp_disabled(compression_type::DEFLATE, {2, 5, 0, true, true, 7}));
  // version 2.5 required
  EXPECT_TRUE(decomp_disabled(compression_type::DEFLATE, {2, 4, 0, true, true, 7}));
  // all integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::DEFLATE, {2, 5, 0, false, true, 7}));

  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {2, 4, 0, true, true, 7}));
  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {2, 3, 2, false, true, 6}));
  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {2, 3, 0, true, true, 6}));
  // 2.3.1 and earlier requires all integrations to be enabled
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {2, 3, 1, false, true, 7}));
  // 2.3 version required
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {2, 2, 0, true, true, 7}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {2, 4, 0, false, false, 7}));
  // 2.4.0 disabled on Pascal
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {2, 4, 0, true, true, 6}));

  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {2, 4, 0, true, true, 7}));
  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {2, 3, 0, false, true, 7}));
  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {2, 2, 0, false, true, 7}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::SNAPPY, {2, 2, 0, false, false, 7}));
}

CUDF_TEST_PROGRAM_MAIN()
