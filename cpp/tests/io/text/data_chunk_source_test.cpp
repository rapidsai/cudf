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

CUDF_TEST_PROGRAM_MAIN()
