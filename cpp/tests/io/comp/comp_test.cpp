/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "io/comp/comp.hpp"
#include "io/comp/gpuinflate.hpp"
#include "io/comp/io_uncomp.hpp"
#include "io/utilities/hostdevice_vector.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <src/io/comp/nvcomp_adapter.hpp>

#include <vector>

using cudf::device_span;
using cudf::io::detail::compression_result;
using cudf::io::detail::compression_status;
namespace nvcomp = cudf::io::detail::nvcomp;

enum class hw { CPU, GPU };

/**
 * @brief Base test fixture for decompression
 *
 * Calls into Decompressor fixture to dispatch actual decompression work,
 * whose interface and setup is different for each codec.
 */
template <typename Decompressor>
struct DecompressTest : public cudf::test::BaseFixture, public testing::WithParamInterface<hw> {
  [[nodiscard]] std::vector<uint8_t> vector_from_string(std::string const str) const
  {
    return {reinterpret_cast<uint8_t const*>(str.c_str()),
            reinterpret_cast<uint8_t const*>(str.c_str()) + strlen(str.c_str())};
  }

  std::vector<uint8_t> Decompress(hw type,
                                  cudf::host_span<uint8_t const> compressed,
                                  size_t uncompressed_size)
  {
    if (type == hw::GPU) {
      if constexpr (has_gpu_impl<Decompressor>::value) {
        return DeviceDecompress(compressed, uncompressed_size);
      } else {
        CUDF_FAIL("Device decompression has not been implemented");
      }
    } else {
      if constexpr (has_cpu_impl<Decompressor>::value) {
        return HostDecompress(compressed, uncompressed_size);
      } else {
        CUDF_FAIL("Host decompression has not been implemented");
      }
    }
  }

  std::vector<uint8_t> DeviceDecompress(cudf::host_span<uint8_t const> compressed,
                                        size_t uncompressed_size)
  {
    auto stream = cudf::get_default_stream();
    std::vector<uint8_t> decompressed(uncompressed_size);
    rmm::device_buffer src{compressed.data(), compressed.size(), stream};
    rmm::device_uvector<uint8_t> dst{decompressed.size(), stream};

    cudf::detail::hostdevice_vector<device_span<uint8_t const>> inf_in(1, stream);
    inf_in[0] = {static_cast<uint8_t const*>(src.data()), src.size()};
    inf_in.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<device_span<uint8_t>> inf_out(1, stream);
    inf_out[0] = dst;
    inf_out.host_to_device_async(stream);

    cudf::detail::hostdevice_vector<compression_result> inf_stat(1, stream);
    inf_stat[0] = {};
    inf_stat.host_to_device_async(stream);

    static_cast<Decompressor*>(this)->device_dispatch(inf_in, inf_out, inf_stat);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      decompressed.data(), dst.data(), dst.size(), cudaMemcpyDefault, stream.value()));
    inf_stat.device_to_host_sync(stream);
    CUDF_EXPECTS(inf_stat[0].status == compression_status::SUCCESS,
                 "Failure in device decompression");

    return decompressed;
  }

  std::vector<uint8_t> HostDecompress(cudf::host_span<uint8_t const> compressed,
                                      size_t uncompressed_size)
  {
    return static_cast<Decompressor*>(this)->host_dispatch(compressed, uncompressed_size);
  }

  template <typename T, typename = void>
  struct has_gpu_impl : std::false_type {};

  template <typename T>
  struct has_gpu_impl<T, std::void_t<decltype(&T::device_dispatch)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_cpu_impl : std::false_type {};

  template <typename T>
  struct has_cpu_impl<T, std::void_t<decltype(&T::host_dispatch)>> : std::true_type {};
};

struct HostCompressTest : public cudf::test::BaseFixture {
  HostCompressTest() { setenv("LIBCUDF_HOST_COMPRESSION", "ON", 1); }
  ~HostCompressTest() override { unsetenv("LIBCUDF_HOST_COMPRESSION"); }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct GzipDecompressTest : public DecompressTest<GzipDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<compression_result> d_inf_stat)
  {
    cudf::io::detail::gpuinflate(d_inf_in,
                                 d_inf_out,
                                 d_inf_stat,
                                 cudf::io::detail::gzip_header_included::YES,
                                 cudf::get_default_stream());
  }

  std::vector<uint8_t> host_dispatch(cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::GZIP, compressed),
                 "Underestimating uncompressed size!");
    return cudf::io::detail::decompress(cudf::io::compression_type::GZIP, compressed);
  }
};

/**
 * @brief Derived fixture for GZIP decompression
 */
struct ZstdDecompressTest : public DecompressTest<ZstdDecompressTest> {
  std::vector<uint8_t> host_dispatch(cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::ZSTD, compressed),
                 "Underestimating uncompressed size!");
    return cudf::io::detail::decompress(cudf::io::compression_type::ZSTD, compressed);
  }
};

/**
 * @brief Derived fixture for Snappy decompression
 */
struct SnappyDecompressTest : public DecompressTest<SnappyDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<compression_result> d_inf_stat)
  {
    cudf::io::detail::gpu_unsnap(d_inf_in, d_inf_out, d_inf_stat, cudf::get_default_stream());
  }

  std::vector<uint8_t> host_dispatch(cudf::host_span<uint8_t const> compressed,
                                     size_t uncompressed_size)
  {
    CUDF_EXPECTS(uncompressed_size <= cudf::io::detail::get_uncompressed_size(
                                        cudf::io::compression_type::SNAPPY, compressed),
                 "Underestimating uncompressed size!");
    return cudf::io::detail::decompress(cudf::io::compression_type::SNAPPY, compressed);
  }
};

/**
 * @brief Derived fixture for Brotli decompression
 */
struct BrotliDecompressTest : public DecompressTest<BrotliDecompressTest> {
  void device_dispatch(device_span<device_span<uint8_t const>> d_inf_in,
                       device_span<device_span<uint8_t>> d_inf_out,
                       device_span<compression_result> d_inf_stat)
  {
    rmm::device_buffer d_scratch{cudf::io::detail::get_gpu_debrotli_scratch_size(1),
                                 cudf::get_default_stream()};

    cudf::io::detail::gpu_debrotli(d_inf_in,
                                   d_inf_out,
                                   d_inf_stat,
                                   d_scratch.data(),
                                   d_scratch.size(),
                                   cudf::get_default_stream());
  }
};

INSTANTIATE_TEST_CASE_P(GzipDecompressTest,
                        GzipDecompressTest,
                        ::testing::Values(hw::CPU, hw::GPU));

TEST_P(GzipDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 31> compressed{
    0x1f, 0x8b, 0x8,  0x0,  0x9,  0x63, 0x99, 0x5c, 0x2,  0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57,
    0x28, 0xcf, 0x2f, 0xca, 0x49, 0x1,  0x0,  0x85, 0x11, 0x4a, 0xd,  0xb,  0x0,  0x0,  0x0};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

INSTANTIATE_TEST_CASE_P(SnappyDecompressTest,
                        SnappyDecompressTest,
                        ::testing::Values(hw::CPU, hw::GPU));

TEST_P(SnappyDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 13> compressed = {
    0xb, 0x28, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

TEST_P(SnappyDecompressTest, ShortLiteralAfterLongCopyAtStartup)
{
  std::string const uncompressed{"Aaaaaaaaaaaah!"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 10> compressed = {
    14, 0x0, 'A', 0x0, 'a', (10 - 4) * 4 + 1, 1, 0x4, 'h', '!'};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

INSTANTIATE_TEST_CASE_P(BrotliDecompressTest, BrotliDecompressTest, ::testing::Values(hw::GPU));

TEST_P(BrotliDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  // NOLINTBEGIN
  constexpr std::array<uint8_t, 15> compressed = {
    0xb, 0x5, 0x80, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64, 0x3};
  // NOLINTEND

  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto output                = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

INSTANTIATE_TEST_CASE_P(ZstdDecompressTest, ZstdDecompressTest, ::testing::Values(hw::CPU));

TEST_P(ZstdDecompressTest, HelloWorld)
{
  std::string const uncompressed{"hello world"};
  std::vector<uint8_t> input = vector_from_string(uncompressed);
  auto compressed =
    cudf::io::detail::compress(cudf::io::compression_type::ZSTD, input, cudf::get_default_stream());
  auto output = Decompress(
    GetParam(), cudf::host_span<uint8_t const>(compressed.data(), compressed.size()), input.size());
  EXPECT_EQ(output, input);
}

struct NvcompConfigTest : public cudf::test::BaseFixture {};

TEST_F(NvcompConfigTest, Compression)
{
  using nvcomp::compression_type;
  auto const& comp_disabled = nvcomp::is_compression_disabled;

  EXPECT_FALSE(comp_disabled(compression_type::DEFLATE, {true, true}));
  // all integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::DEFLATE, {false, true}));

  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {true, true}));
  EXPECT_FALSE(comp_disabled(compression_type::ZSTD, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::ZSTD, {false, false}));

  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {true, true}));
  EXPECT_FALSE(comp_disabled(compression_type::SNAPPY, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(comp_disabled(compression_type::SNAPPY, {false, false}));
}

TEST_F(NvcompConfigTest, Decompression)
{
  using nvcomp::compression_type;
  auto const& decomp_disabled = nvcomp::is_decompression_disabled;

  EXPECT_FALSE(decomp_disabled(compression_type::DEFLATE, {true, true}));
  // all integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::DEFLATE, {false, true}));

  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {true, true}));
  EXPECT_FALSE(decomp_disabled(compression_type::ZSTD, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::ZSTD, {false, false}));

  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {true, true}));
  EXPECT_FALSE(decomp_disabled(compression_type::SNAPPY, {false, true}));
  // stable integrations enabled required
  EXPECT_TRUE(decomp_disabled(compression_type::SNAPPY, {false, false}));
}

TEST_F(HostCompressTest, SnappyCompression)
{
  std::vector<uint8_t> expected;
  expected.reserve(8 * (32 << 20));
  for (size_t size = 1; size < 32 << 20; size *= 2) {
    // Using number strings to generate data that is compressible, but not trivially so
    for (size_t i = size / 2; i < size; ++i) {
      auto const num_string = std::to_string(i);
      // Keep adding to the test data
      expected.insert(expected.end(), num_string.begin(), num_string.end());
    }
    auto const compressed = cudf::io::detail::compress(
      cudf::io::compression_type::SNAPPY, expected, cudf::get_default_stream());
    auto const decompressed =
      cudf::io::detail::decompress(cudf::io::compression_type::SNAPPY, compressed);
    EXPECT_EQ(expected, decompressed);
  }
}

CUDF_TEST_PROGRAM_MAIN()
