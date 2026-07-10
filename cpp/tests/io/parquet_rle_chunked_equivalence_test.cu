/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../src/io/parquet/rle_stream.cuh"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int decode_threads = 128;
constexpr int max_values     = 1024;

template <typename level_t>
__global__ void decode_compare_kernel(uint8_t const* encoded,
                                      int encoded_size,
                                      int level_bits,
                                      int total_values,
                                      int first_count,
                                      level_t* ring_output,
                                      level_t* chunked_output,
                                      int* decoded)
{
  using namespace cudf::io::parquet::detail;
  __shared__ rle_run ring_runs[rle_stream_required_run_buffer_size<decode_threads>()];
  __shared__ rle_run chunked_runs[rle_stream_required_run_buffer_size<decode_threads>()];

  rle_stream<level_t, decode_threads, max_values> ring{ring_runs};
  rle_stream_chunked<level_t, decode_threads, max_values> chunked{chunked_runs};

  int const t = threadIdx.x;
  ring.init(level_bits, encoded, encoded + encoded_size, ring_output, total_values);
  chunked.init(level_bits, encoded, encoded + encoded_size, chunked_output, total_values);

  int ring_decoded    = ring.decode_next(t, first_count);
  int chunked_decoded = chunked.decode_next(t, first_count);
  if (first_count < total_values) {
    ring_decoded += ring.decode_next(t, total_values - first_count);
    chunked_decoded += chunked.decode_next(t, total_values - first_count);
  }

  if (t == 0) {
    decoded[0] = ring_decoded;
    decoded[1] = chunked_decoded;
  }
}

void append_vlq(std::vector<uint8_t>& out, uint32_t value)
{
  while (value >= 0x80) {
    out.push_back(static_cast<uint8_t>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  out.push_back(static_cast<uint8_t>(value));
}

void append_repeated(std::vector<uint8_t>& out, int count, uint32_t value, int level_bits)
{
  append_vlq(out, static_cast<uint32_t>(count) << 1);
  int const width = (level_bits + 7) >> 3;
  for (int byte = 0; byte < width; ++byte) {
    out.push_back(static_cast<uint8_t>((value >> (byte * 8)) & 0xff));
  }
}

void append_literal(std::vector<uint8_t>& out, std::vector<uint32_t> const& values, int level_bits)
{
  append_vlq(out, (static_cast<uint32_t>(values.size() / 8) << 1) | 1u);
  std::vector<uint8_t> payload((values.size() * level_bits + 7) / 8);
  for (std::size_t i = 0; i < values.size(); ++i) {
    int const bit = static_cast<int>(i) * level_bits;
    for (int b = 0; b < level_bits; ++b) {
      if ((values[i] >> b) & 1u) { payload[(bit + b) >> 3] |= uint8_t{1} << ((bit + b) & 7); }
    }
  }
  out.insert(out.end(), payload.begin(), payload.end());
}

template <typename level_t>
void run_case(std::vector<uint8_t> const& encoded,
              int level_bits,
              int total_values,
              int first_count)
{
  auto stream = cudf::get_default_stream();
  rmm::device_uvector<uint8_t> d_encoded(encoded.size(), stream);
  rmm::device_uvector<level_t> d_ring(total_values, stream);
  rmm::device_uvector<level_t> d_chunked(total_values, stream);
  rmm::device_uvector<int> d_decoded(2, stream);

  CUDF_CUDA_TRY(cudaMemcpyAsync(d_encoded.data(),
                                encoded.data(),
                                encoded.size() * sizeof(uint8_t),
                                cudaMemcpyDefault,
                                stream.value()));
  CUDF_CUDA_TRY(
    cudaMemsetAsync(d_ring.data(), 0xff, total_values * sizeof(level_t), stream.value()));
  CUDF_CUDA_TRY(
    cudaMemsetAsync(d_chunked.data(), 0xff, total_values * sizeof(level_t), stream.value()));

  decode_compare_kernel<<<1, decode_threads, 0, stream.value()>>>(d_encoded.data(),
                                                                  encoded.size(),
                                                                  level_bits,
                                                                  total_values,
                                                                  first_count,
                                                                  d_ring.data(),
                                                                  d_chunked.data(),
                                                                  d_decoded.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  std::vector<level_t> ring(total_values);
  std::vector<level_t> chunked(total_values);
  std::vector<int> decoded(2);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    ring.data(), d_ring.data(), total_values * sizeof(level_t), cudaMemcpyDefault, stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(chunked.data(),
                                d_chunked.data(),
                                total_values * sizeof(level_t),
                                cudaMemcpyDefault,
                                stream.value()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(decoded.data(),
                                d_decoded.data(),
                                decoded.size() * sizeof(int),
                                cudaMemcpyDefault,
                                stream.value()));
  stream.synchronize();

  EXPECT_EQ(decoded[0], total_values);
  EXPECT_EQ(decoded[1], total_values);
  EXPECT_EQ(ring, chunked);
}

class ParquetRleChunkedEquivalenceTest : public cudf::test::BaseFixture {};

TEST_F(ParquetRleChunkedEquivalenceTest, SingleRepeated)
{
  std::vector<uint8_t> encoded;
  append_repeated(encoded, 5, 3, 4);
  run_case<uint8_t>(encoded, 4, 5, 5);
}

TEST_F(ParquetRleChunkedEquivalenceTest, SingleLiteral)
{
  std::vector<uint8_t> encoded;
  append_literal(encoded, {0, 1, 2, 3, 4, 5, 6, 7}, 4);
  run_case<uint8_t>(encoded, 4, 8, 8);
}

TEST_F(ParquetRleChunkedEquivalenceTest, MixedRuns)
{
  std::vector<uint8_t> encoded;
  append_repeated(encoded, 3, 2, 4);
  append_literal(encoded, {1, 3, 5, 7, 9, 11, 13, 15}, 4);
  append_repeated(encoded, 4, 6, 4);
  run_case<uint8_t>(encoded, 4, 15, 15);
}

TEST_F(ParquetRleChunkedEquivalenceTest, ManyShortRepeatedRuns)
{
  std::vector<uint8_t> encoded;
  for (int i = 0; i < 130; ++i) {
    append_repeated(encoded, 1, i & 15, 4);
  }
  run_case<uint8_t>(encoded, 4, 130, 130);
}

TEST_F(ParquetRleChunkedEquivalenceTest, PartialDecode)
{
  std::vector<uint8_t> encoded;
  append_repeated(encoded, 3, 2, 4);
  append_literal(encoded, {1, 3, 5, 7, 9, 11, 13, 15}, 4);
  append_repeated(encoded, 4, 6, 4);
  run_case<uint8_t>(encoded, 4, 15, 4);
}

TEST_F(ParquetRleChunkedEquivalenceTest, LevelBits8Uint8)
{
  std::vector<uint8_t> encoded;
  append_literal(encoded, {0, 17, 34, 51, 68, 85, 102, 119}, 8);
  run_case<uint8_t>(encoded, 8, 8, 8);
}

TEST_F(ParquetRleChunkedEquivalenceTest, LevelBits8Uint16)
{
  std::vector<uint8_t> encoded;
  append_repeated(encoded, 5, 200, 8);
  append_literal(encoded, {0, 25, 50, 75, 100, 125, 150, 175}, 8);
  run_case<uint16_t>(encoded, 8, 13, 13);
}

}  // namespace
