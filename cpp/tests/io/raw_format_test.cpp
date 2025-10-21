/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/raw_format.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>

#include <fstream>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

/**
 * @brief Base test fixture for raw format write/read tests
 */
struct RawFormatTest : public cudf::test::BaseFixture {
  /**
   * @brief Test write and read roundtrip using file-based sink/source
   */
  void run_test_file(cudf::table_view const& expected)
  {
    auto const filepath = temp_env->get_temp_filepath("test.raw");

    // Write table to file
    cudf::io::write_raw(expected, cudf::io::sink_info{filepath});

    // Read table back from file
    auto result = cudf::io::read_raw(cudf::io::source_info{filepath});

    // Verify the tables match
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }

  /**
   * @brief Test write and read roundtrip using memory buffer sink/source
   */
  void run_test_buffer(cudf::table_view const& expected)
  {
    std::vector<char> buffer;

    // Write to memory buffer
    cudf::io::write_raw(expected, cudf::io::sink_info{&buffer});

    // Read from memory buffer
    auto host_buffer = cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
    auto result = cudf::io::read_raw(cudf::io::source_info{host_buffer});

    // Verify the tables match
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }

  /**
   * @brief Run both file and buffer tests
   */
  void run_test(cudf::table_view const& t)
  {
    run_test_file(t);
    run_test_buffer(t);
  }
};

// ============================================================================
// Basic data type tests
// ============================================================================

TEST_F(RawFormatTest, SingleColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7},
                                                      {true, true, true, false, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, SingleColumnFixedWidthNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, MultiColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> col2({7, 8, 6, 5, 4, 3, 2},
                                                     {true, false, true, true, true, true, true});
  cudf::test::fixed_width_column_wrapper<double> col3({8, 4, 2, 0, 7, 1, 3},
                                                      {false, true, true, true, true, true, true});

  auto const expected = cudf::table_view{{col1, col2, col3}};
  run_test(expected);
}

TEST_F(RawFormatTest, MultiColumnWithStrings)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::strings_column_wrapper col2({"Lorem", "ipsum", "dolor", "sit", "amet", "ort", "ral"},
                                          {true, false, true, true, true, false, true});
  cudf::test::strings_column_wrapper col3({"", "this", "is", "a", "column", "of", "strings"});

  auto const expected = cudf::table_view{{col1, col2, col3}};
  run_test(expected);
}

TEST_F(RawFormatTest, EmptyTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, SingleRow)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({42});
  cudf::test::strings_column_wrapper str_col({"hello"});

  auto const expected = cudf::table_view{{col, str_col}};
  run_test(expected);
}

// ============================================================================
// Nested type tests
// ============================================================================

TEST_F(RawFormatTest, ListsOfIntegers)
{
  cudf::test::lists_column_wrapper<int32_t> col{{1, 2, 3}, {4, 5}, {}, {6, 7, 8, 9}, {10}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, ListsWithNulls)
{
  // Create a list column with element validity
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  cudf::test::lists_column_wrapper<int32_t> col{
    {{1, 2, 3}, valids}, {4, 5}, {}, {{6, 7, 8, 9}, valids}, {10}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, StructColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3, 4};
  cudf::test::strings_column_wrapper col2{"a", "b", "c", "d"};
  cudf::test::structs_column_wrapper struct_col{{col1, col2}};

  auto const expected = cudf::table_view{{struct_col}};
  run_test(expected);
}

TEST_F(RawFormatTest, StructWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{{1, 2, 3, 4}, {true, false, true, true}};
  cudf::test::strings_column_wrapper col2{{"a", "b", "c", "d"}, {true, true, false, true}};
  cudf::test::structs_column_wrapper struct_col{{col1, col2}, {true, true, true, false}};

  auto const expected = cudf::table_view{{struct_col}};
  run_test(expected);
}

// ============================================================================
// Edge case tests
// ============================================================================

TEST_F(RawFormatTest, LargeTable)
{
  constexpr int num_rows = 100000;

  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col1(sequence, sequence + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col2(sequence, sequence + num_rows);

  auto const expected = cudf::table_view{{col1, col2}};
  run_test(expected);
}

TEST_F(RawFormatTest, AllNullColumn)
{
  auto all_nulls = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5}, all_nulls);

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, MixedNullability)
{
  cudf::test::fixed_width_column_wrapper<int32_t> nullable_col({1, 2, 3, 4, 5},
                                                               {true, false, true, false, true});
  cudf::test::fixed_width_column_wrapper<float> non_nullable_col({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  cudf::test::strings_column_wrapper string_col({"a", "b", "c", "d", "e"},
                                                {true, true, false, true, true});

  auto const expected = cudf::table_view{{nullable_col, non_nullable_col, string_col}};
  run_test(expected);
}

// ============================================================================
// Error handling tests
// ============================================================================

TEST_F(RawFormatTest, InvalidHeaderMagic)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("invalid.raw");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::write_raw(expected, cudf::io::sink_info{filepath});

  // Corrupt the magic number
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  uint32_t bad_magic = 0xDEADBEEF;
  file.write(reinterpret_cast<char*>(&bad_magic), sizeof(uint32_t));
  file.close();

  // Reading should fail
  EXPECT_THROW(cudf::io::read_raw(cudf::io::source_info{filepath}), cudf::logic_error);
}

TEST_F(RawFormatTest, InvalidHeaderVersion)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("invalid_version.raw");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::write_raw(expected, cudf::io::sink_info{filepath});

  // Corrupt the version
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(uint32_t));  // Skip magic
  uint32_t bad_version = 999;
  file.write(reinterpret_cast<char*>(&bad_version), sizeof(uint32_t));
  file.close();

  // Reading should fail
  EXPECT_THROW(cudf::io::read_raw(cudf::io::source_info{filepath}), cudf::logic_error);
}

TEST_F(RawFormatTest, TruncatedFile)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("truncated.raw");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};
  cudf::io::write_raw(expected, cudf::io::sink_info{filepath});

  // Truncate the file
  std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
  file.write("TRUNCATED", 9);
  file.close();

  // Reading should fail
  EXPECT_THROW(cudf::io::read_raw(cudf::io::source_info{filepath}), cudf::logic_error);
}

// ============================================================================
// Special data types
// ============================================================================

TEST_F(RawFormatTest, BooleanColumn)
{
  cudf::test::fixed_width_column_wrapper<bool> col({true, false, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, TimestampColumn)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_ms> col{cudf::timestamp_ms{0ms},
                                                     cudf::timestamp_ms{100ms},
                                                     cudf::timestamp_ms{200ms},
                                                     cudf::timestamp_ms{300ms}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(RawFormatTest, DurationColumn)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_ns> col{cudf::duration_ns{0ns},
                                                    cudf::duration_ns{1000ns},
                                                    cudf::duration_ns{2000ns},
                                                    cudf::duration_ns{3000ns}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

CUDF_TEST_PROGRAM_MAIN()
