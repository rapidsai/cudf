/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <cudf/io/text/data_chunk_source_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <fstream>
#include <random>

using namespace cudf::test;

auto const temp_env = static_cast<TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));

struct DataChunkSourceTest : public BaseFixture {
};

std::string chunk_to_host(const cudf::io::text::device_data_chunk& chunk)
{
  std::string result(chunk.size(), '\0');
  cudaMemcpy(result.data(), chunk.data(), chunk.size(), cudaMemcpyDeviceToHost);
  return result;
}

void test_source(const std::string& content, const cudf::io::text::data_chunk_source& source)
{
  {
    // full contents
    auto reader = source.create_reader();
    auto chunk  = reader->get_next_chunk(content.size(), rmm::cuda_stream_default);
    ASSERT_EQ(chunk->size(), content.size());
    ASSERT_EQ(chunk_to_host(*chunk), content);
  }
  {
    // skipping contents
    auto reader = source.create_reader();
    reader->skip_bytes(4);
    auto chunk = reader->get_next_chunk(content.size(), rmm::cuda_stream_default);
    ASSERT_EQ(chunk->size(), content.size() - 4);
    ASSERT_EQ(chunk_to_host(*chunk), content.substr(4));
  }
  {
    // reading multiple chunks
    auto reader = source.create_reader();
    auto chunk1 = reader->get_next_chunk(content.size() / 2, rmm::cuda_stream_default);
    auto chunk2 =
      reader->get_next_chunk(content.size() - content.size() / 2, rmm::cuda_stream_default);
    ASSERT_EQ(chunk1->size(), content.size() / 2);
    ASSERT_EQ(chunk2->size(), content.size() - content.size() / 2);
    ASSERT_EQ(chunk_to_host(*chunk1), content.substr(0, content.size() / 2));
    ASSERT_EQ(chunk_to_host(*chunk2), content.substr(content.size() / 2));
  }
  {
    // reading too many bytes
    auto reader = source.create_reader();
    auto chunk  = reader->get_next_chunk(content.size() + 10, rmm::cuda_stream_default);
    ASSERT_EQ(chunk->size(), content.size());
    ASSERT_EQ(chunk_to_host(*chunk), content);
    auto next_chunk = reader->get_next_chunk(1, rmm::cuda_stream_default);
    ASSERT_EQ(next_chunk->size(), 0);
  }
  {
    // skipping past the end
    auto reader = source.create_reader();
    reader->skip_bytes(content.size() + 10);
    auto next_chunk = reader->get_next_chunk(1, rmm::cuda_stream_default);
    ASSERT_EQ(next_chunk->size(), 0);
  }
}

TEST_F(DataChunkSourceTest, Device)
{
  std::string content = "device buffer source";
  cudf::string_scalar scalar(content);
  auto source = cudf::io::text::make_source(scalar);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, File)
{
  std::string content = "file source";
  auto filename       = temp_env->get_temp_filepath("file_source");
  {
    std::ofstream file{filename};
    file << content;
  }
  auto source = cudf::io::text::make_source_from_file(filename);

  test_source(content, *source);
}

TEST_F(DataChunkSourceTest, Host)
{
  std::string content = "host buffer source";
  auto source         = cudf::io::text::make_source(content);

  test_source(content, *source);
}

template <typename T>
void write_int(std::ostream& stream, T val)
{
  // endianness check
  auto const i = static_cast<T>(0x0100);
  std::array<char, sizeof(T)> bytes;
  std::memcpy(&bytes[0], &i, sizeof(T));
  auto const little_endian = bytes[0] == 0;
  // write
  std::memcpy(&bytes[0], &val, sizeof(T));
  if (not little_endian) { std::reverse(bytes.begin(), bytes.end()); }
  stream.write(bytes.data(), bytes.size());
}

void write_bgzip_block(std::ostream& stream,
                       const std::string& data,
                       bool add_extra_garbage_before,
                       bool add_extra_garbage_after)
{
  std::array<uint8_t, 10> const header{{
    31,   // magic number
    139,  // magic number
    8,    // compression type: deflate
    4,    // flags: extra header
    0,    // mtime
    0,    // mtime
    0,    // mtime
    0,    // mtime: irrelevant
    4,    // xfl: irrelevant
    3     // OS: irrelevant
  }};
  std::array<char, 4> extra_blocklen_field{{66, 67, 2, 0}};
  std::array<char, 11> extra_garbage_field1{{13,  // magic number
                                             37,  // magic number
                                             7,   // field length
                                             0,   // field length
                                             1,
                                             2,
                                             3,
                                             4,
                                             5,
                                             6,
                                             7}};
  std::array<char, 23> extra_garbage_field2{{12,  // magic number
                                             34,  // magic number
                                             2,   // field length
                                             0,   // field length
                                             1,  2,
                                             56,  // magic number
                                             78,  // magic number
                                             1,   // field length
                                             0,   // field length
                                             3,   //
                                             90,  // magic number
                                             12,  // magic number
                                             8,   // field length
                                             0,   // field length
                                             1,  2, 3, 4, 5, 6, 7, 8}};
  stream.write(reinterpret_cast<const char*>(header.data()), header.size());
  uint16_t extra_size = extra_blocklen_field.size() + 2;
  if (add_extra_garbage_before) { extra_size += extra_garbage_field1.size(); }
  if (add_extra_garbage_after) { extra_size += extra_garbage_field2.size(); }
  write_int(stream, extra_size);
  if (add_extra_garbage_before) {
    stream.write(extra_garbage_field1.data(), extra_garbage_field1.size());
  }
  stream.write(extra_blocklen_field.data(), extra_blocklen_field.size());
  auto compressed_size          = data.size() + 5;
  uint16_t block_size_minus_one = compressed_size + 19 + extra_size;
  write_int(stream, block_size_minus_one);
  if (add_extra_garbage_after) {
    stream.write(extra_garbage_field2.data(), extra_garbage_field2.size());
  }
  write_int<uint8_t>(stream, 1);
  write_int<uint16_t>(stream, data.size());
  write_int<uint16_t>(stream, ~static_cast<uint16_t>(data.size()));
  stream.write(data.data(), data.size());
  // this does not produce a valid file, since we write 0 as the CRC
  // the parser ignores the checksum, so it doesn't matter to the test
  // to check output with gzip, plug in the CRC of `data` here.
  write_int<uint32_t>(stream, 0);
  write_int<uint32_t>(stream, data.size());
}

void write_bgzip(std::ostream& stream,
                 const std::string& data,
                 std::default_random_engine& rng,
                 bool write_eof = true)
{
  // make sure the block size with header stays below 65536
  std::uniform_int_distribution<std::size_t> block_size_dist{1, 65000};
  auto begin     = data.begin();
  auto const end = data.end();
  int i          = 0;
  while (begin < end) {
    auto len = std::min<std::size_t>(end - begin, block_size_dist(rng));
    write_bgzip_block(stream, std::string{begin, begin + len}, i & 1, i & 2);
    begin += len;
    i++;
  }
  if (write_eof) { write_bgzip_block(stream, {}, false, false); }
}

TEST_F(DataChunkSourceTest, BgzipSource)
{
  auto filename = temp_env->get_temp_filepath("bgzip_source");
  std::string input{"bananarama"};
  for (int i = 0; i < 24; i++) {
    input = input + input;
  }
  {
    std::ofstream stream{filename};
    std::default_random_engine rng{};
    write_bgzip(stream, input, rng);
  }

  auto source = cudf::io::text::make_source_from_bgzip_file(filename);

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsets)
{
  auto filename = temp_env->get_temp_filepath("bgzip_source");
  std::string input{"bananarama"};
  for (int i = 0; i < 24; i++) {
    input = input + input;
  }
  std::string garbage{"garbage"};
  for (int i = 0; i < 10; i++) {
    garbage = garbage + garbage;
  }
  std::string garbage2{"GARBAGE"};
  std::string begininput{"begin of bananarama"};
  std::string endinput{"end of bananarama"};
  std::size_t begin_compressed_offset{};
  std::size_t end_compressed_offset{};
  std::size_t begin_local_offset{garbage2.size()};
  std::size_t end_local_offset{endinput.size()};
  {
    std::ofstream stream{filename};
    stream.write(garbage.data(), garbage.size());
    std::default_random_engine rng{};
    begin_compressed_offset = stream.tellp();
    write_bgzip_block(stream, garbage2 + begininput, false, false);
    write_bgzip(stream, input, rng, false);
    end_compressed_offset = stream.tellp();
    write_bgzip_block(stream, endinput + garbage2 + garbage2, false, false);
    write_bgzip_block(stream, {}, false, false);
    stream.write(garbage.data(), garbage.size());
  }
  input = begininput + input + endinput;

  auto source =
    cudf::io::text::make_source_from_bgzip_file(filename,
                                                begin_compressed_offset << 16 | begin_local_offset,
                                                end_compressed_offset << 16 | end_local_offset);

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsetsSingleGZipBlock)
{
  auto filename = temp_env->get_temp_filepath("bgzip_source");
  std::string input{"collection unit brings"};
  std::string garbage{"garbage"};
  std::string garbage2{"GARBAGE"};
  std::size_t begin_compressed_offset{};
  std::size_t end_compressed_offset{};
  std::size_t begin_local_offset{garbage.size()};
  std::size_t end_local_offset{garbage.size() + input.size()};
  {
    std::ofstream stream{filename};
    write_bgzip_block(stream, garbage + input + garbage2, false, false);
    write_bgzip_block(stream, {}, false, false);
  }

  auto source =
    cudf::io::text::make_source_from_bgzip_file(filename,
                                                begin_compressed_offset << 16 | begin_local_offset,
                                                end_compressed_offset << 16 | end_local_offset);

  test_source(input, *source);
}

TEST_F(DataChunkSourceTest, BgzipSourceVirtualOffsetsSingleChunk)
{
  auto filename = temp_env->get_temp_filepath("bgzip_source");
  std::string input{"collection unit brings"};
  std::string garbage{"garbage"};
  std::string garbage2{"GARBAGE"};
  std::size_t begin_compressed_offset{};
  std::size_t end_compressed_offset{};
  std::size_t begin_local_offset{garbage.size()};
  std::size_t end_local_offset{input.size() - 10};
  {
    std::ofstream stream{filename};
    write_bgzip_block(stream, garbage + input.substr(0, 10), false, false);
    end_compressed_offset = stream.tellp();
    write_bgzip_block(stream, input.substr(10) + garbage2, false, false);
    write_bgzip_block(stream, {}, false, false);
  }

  auto source =
    cudf::io::text::make_source_from_bgzip_file(filename,
                                                begin_compressed_offset << 16 | begin_local_offset,
                                                end_compressed_offset << 16 | end_local_offset);

  test_source(input, *source);
}

CUDF_TEST_PROGRAM_MAIN()
