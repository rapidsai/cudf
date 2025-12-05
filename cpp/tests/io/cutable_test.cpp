/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/io/cutable.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <fstream>
#include <vector>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

/**
 * @brief Base test fixture for cutable write/read tests
 */
struct CutableTest : public cudf::test::BaseFixture {
  /**
   * @brief Test write and read roundtrip using file-based sink/source
   */
  void run_test_file(cudf::table_view const& expected)
  {
    auto const filepath = temp_env->get_temp_filepath("test.cutable");

    // Write table to file
    cudf::io::experimental::write_cutable(
      cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

    // Read table back from file
    auto result = cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build());

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
    cudf::io::experimental::write_cutable(
      cudf::io::cutable_writer_options::builder(cudf::io::sink_info{&buffer}, expected).build());

    // Read from memory buffer
    auto host_buffer = cudf::host_span<std::byte const>(
      reinterpret_cast<std::byte const*>(buffer.data()), buffer.size());
    auto result = cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{host_buffer}).build());

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

  /**
   * @brief Test write and read roundtrip using device buffer source
   */
  void run_test_device_buffer(cudf::table_view const& expected)
  {
    std::vector<char> buffer;

    // Write to memory buffer first
    cudf::io::experimental::write_cutable(
      cudf::io::cutable_writer_options::builder(cudf::io::sink_info{&buffer}, expected).build());

    // Copy to device buffer
    rmm::device_buffer device_buffer(buffer.size(), cudf::get_default_stream());
    CUDF_CUDA_TRY(
      cudaMemcpy(device_buffer.data(), buffer.data(), buffer.size(), cudaMemcpyHostToDevice));

    // Read from device buffer
    auto device_span = cudf::device_span<std::byte const>(
      static_cast<std::byte const*>(device_buffer.data()), device_buffer.size());
    auto result = cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{device_span}).build());

    // Verify the tables match
    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.table);
  }
};

// ============================================================================
// Basic data type tests
// ============================================================================

TEST_F(CutableTest, SingleColumnFixedWidth)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7},
                                                      {true, true, true, false, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, SingleColumnFixedWidthNonNullable)
{
  cudf::test::fixed_width_column_wrapper<int64_t> col({1, 2, 3, 4, 5, 6, 7});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, MultiColumnFixedWidth)
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

TEST_F(CutableTest, MultiColumnWithStrings)
{
  cudf::test::fixed_width_column_wrapper<int16_t> col1(
    {1, 2, 3, 4, 5, 6, 7}, {true, true, true, false, true, false, true});
  cudf::test::strings_column_wrapper col2({"Lorem", "ipsum", "dolor", "sit", "amet", "ort", "ral"},
                                          {true, false, true, true, true, false, true});
  cudf::test::strings_column_wrapper col3({"", "this", "is", "a", "column", "of", "strings"});

  auto const expected = cudf::table_view{{col1, col2, col3}};
  run_test(expected);
}

TEST_F(CutableTest, EmptyTable)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, SingleRow)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({42});
  cudf::test::strings_column_wrapper str_col({"hello"});

  auto const expected = cudf::table_view{{col, str_col}};
  run_test(expected);
}

// ============================================================================
// Nested type tests
// ============================================================================

TEST_F(CutableTest, ListsOfIntegers)
{
  cudf::test::lists_column_wrapper<int32_t> col{{1, 2, 3}, {4, 5}, {}, {6, 7, 8, 9}, {10}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, ListsWithNulls)
{
  // Create a list column with element validity
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
  cudf::test::lists_column_wrapper<int32_t> col{
    {{1, 2, 3}, valids}, {4, 5}, {}, {{6, 7, 8, 9}, valids}, {10}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, StructColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col1{1, 2, 3, 4};
  cudf::test::strings_column_wrapper col2{"a", "b", "c", "d"};
  cudf::test::structs_column_wrapper struct_col{{col1, col2}};

  auto const expected = cudf::table_view{{struct_col}};
  run_test(expected);
}

TEST_F(CutableTest, StructWithNulls)
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

TEST_F(CutableTest, LargeTable)
{
  constexpr int num_rows = 100000;

  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<int32_t> col1(sequence, sequence + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col2(sequence, sequence + num_rows);

  auto const expected = cudf::table_view{{col1, col2}};
  run_test(expected);
}

TEST_F(CutableTest, AllNullColumn)
{
  auto all_nulls = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return false; });
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5}, all_nulls);

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, MixedNullability)
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

TEST_F(CutableTest, InvalidHeaderMagic)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("invalid.cutable");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

  // Corrupt the magic number
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  uint32_t bad_magic = 0xDEADBEEF;
  file.write(reinterpret_cast<char*>(&bad_magic), sizeof(uint32_t));
  file.close();

  // Reading should fail
  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

TEST_F(CutableTest, InvalidHeaderVersion)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("invalid_version.cutable");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

  // Corrupt the version
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(uint32_t));  // Skip magic
  uint32_t bad_version = 999;
  file.write(reinterpret_cast<char*>(&bad_version), sizeof(uint32_t));
  file.close();

  // Reading should fail
  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

TEST_F(CutableTest, TruncatedFile)
{
  // Create a valid file first
  auto const filepath = temp_env->get_temp_filepath("truncated.cutable");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

  // Truncate the file
  std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
  file.write("TRUNCATED", 9);
  file.close();

  // Reading should fail
  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

// ============================================================================
// Special data types
// ============================================================================

TEST_F(CutableTest, BooleanColumn)
{
  cudf::test::fixed_width_column_wrapper<bool> col({true, false, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, TimestampColumn)
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

TEST_F(CutableTest, DurationColumn)
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

// ============================================================================
// Unsigned integer types
// ============================================================================

TEST_F(CutableTest, Uint8Column)
{
  cudf::test::fixed_width_column_wrapper<uint8_t> col({0, 1, 255, 128, 64},
                                                      {true, true, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Uint16Column)
{
  cudf::test::fixed_width_column_wrapper<uint16_t> col({0, 1, 65535, 32768, 16384},
                                                       {true, true, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Uint32Column)
{
  std::vector<uint32_t> data{0, 1, 4294967295U, 2147483648U, 1073741824U};
  std::vector<bool> validity{true, true, true, false, true};
  cudf::test::fixed_width_column_wrapper<uint32_t> col(data.begin(), data.end(), validity.begin());

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Uint64Column)
{
  cudf::test::fixed_width_column_wrapper<uint64_t> col(
    {0ULL, 1ULL, 18446744073709551615ULL, 9223372036854775808ULL, 4611686018427387904ULL},
    {true, true, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Int8Column)
{
  cudf::test::fixed_width_column_wrapper<int8_t> col({-128, -64, 0, 64, 127},
                                                     {true, true, false, true, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

// ============================================================================
// All timestamp resolutions
// ============================================================================

TEST_F(CutableTest, TimestampDays)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_D> col{cudf::timestamp_D{cudf::duration_D{0}},
                                                    cudf::timestamp_D{cudf::duration_D{100}},
                                                    cudf::timestamp_D{cudf::duration_D{200}},
                                                    cudf::timestamp_D{cudf::duration_D{300}}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, TimestampSeconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_s> col{cudf::timestamp_s{0s},
                                                    cudf::timestamp_s{100s},
                                                    cudf::timestamp_s{200s},
                                                    cudf::timestamp_s{300s}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, TimestampMicroseconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_us> col{cudf::timestamp_us{0us},
                                                     cudf::timestamp_us{100us},
                                                     cudf::timestamp_us{200us},
                                                     cudf::timestamp_us{300us}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, TimestampNanoseconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::timestamp_ns> col{cudf::timestamp_ns{0ns},
                                                     cudf::timestamp_ns{100ns},
                                                     cudf::timestamp_ns{200ns},
                                                     cudf::timestamp_ns{300ns}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

// ============================================================================
// All duration resolutions
// ============================================================================

TEST_F(CutableTest, DurationDays)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_D> col{
    cudf::duration_D{0}, cudf::duration_D{100}, cudf::duration_D{200}, cudf::duration_D{300}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, DurationSeconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_s> col{
    cudf::duration_s{0s}, cudf::duration_s{100s}, cudf::duration_s{200s}, cudf::duration_s{300s}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, DurationMilliseconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_ms> col{cudf::duration_ms{0ms},
                                                    cudf::duration_ms{100ms},
                                                    cudf::duration_ms{200ms},
                                                    cudf::duration_ms{300ms}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, DurationMicroseconds)
{
  using namespace cudf::test;
  using namespace cuda::std::chrono;

  fixed_width_column_wrapper<cudf::duration_us> col{cudf::duration_us{0us},
                                                    cudf::duration_us{1000us},
                                                    cudf::duration_us{2000us},
                                                    cudf::duration_us{3000us}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

// ============================================================================
// Decimal types
// ============================================================================

TEST_F(CutableTest, Decimal32Column)
{
  using namespace numeric;
  cudf::test::fixed_point_column_wrapper<int32_t> col{
    {12345, -67890, 0, 99999}, {true, true, false, true}, scale_type{2}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Decimal64Column)
{
  using namespace numeric;
  cudf::test::fixed_point_column_wrapper<int64_t> col{
    {123456789012345LL, -987654321098765LL, 0LL}, {true, true, false}, scale_type{-5}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, Decimal128Column)
{
  using namespace numeric;
  // Note: __int128_t values need to be constructed carefully
  cudf::test::fixed_point_column_wrapper<__int128_t> col{
    {0, 0, 0}, {true, true, false}, scale_type{0}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

// ============================================================================
// Dictionary types
// ============================================================================

TEST_F(CutableTest, DictionaryColumn)
{
  cudf::test::strings_column_wrapper keys({"apple", "banana", "cherry", "date"});
  cudf::test::fixed_width_column_wrapper<int32_t> indices{0, 1, 2, 0, 1, 3, 2, 1};

  auto dictionary     = cudf::make_dictionary_column(keys, indices);
  auto const expected = cudf::table_view{{dictionary->view()}};
  run_test(expected);
}

TEST_F(CutableTest, DictionaryColumnWithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> keys{10, 20, 30, 40};
  cudf::test::fixed_width_column_wrapper<int32_t> indices{{0, 1, 2, 0, 1, 3},
                                                          {true, true, false, true, true, true}};

  auto dictionary     = cudf::make_dictionary_column(keys, indices);
  auto const expected = cudf::table_view{{dictionary->view()}};
  run_test(expected);
}

// ============================================================================
// Complex nested types
// ============================================================================

TEST_F(CutableTest, ListsOfLists)
{
  // Create nested lists using make_lists_column
  cudf::test::lists_column_wrapper<int32_t> child_list{{1, 2}, {3, 4}, {5, 6, 7}, {8}, {}, {9, 10}};

  // Create offsets for the outer list: [0, 2, 4, 6] means 3 lists: [0-2), [2-4), [4-6)
  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 6}.release();
  auto lists_col =
    cudf::make_lists_column(3, std::move(offsets), child_list.release(), 0, rmm::device_buffer{});

  auto const expected = cudf::table_view{{lists_col->view()}};
  run_test(expected);
}

TEST_F(CutableTest, ListsOfStrings)
{
  cudf::test::lists_column_wrapper<cudf::string_view> col{
    {{"hello", "world"}, {"foo", "bar", "baz"}, {}, {"test"}}};

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, ListsOfStructs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> struct_col1{1, 2, 3, 4, 5, 6};
  cudf::test::strings_column_wrapper struct_col2{"a", "b", "c", "d", "e", "f"};
  cudf::test::structs_column_wrapper struct_col{{struct_col1, struct_col2}};

  // Create offsets for lists: [0, 3, 6] means 2 lists: [0-3), [3-6)
  auto offsets = cudf::test::fixed_width_column_wrapper<cudf::size_type>{0, 3, 6}.release();
  auto lists_col =
    cudf::make_lists_column(2, std::move(offsets), struct_col.release(), 0, rmm::device_buffer{});

  auto const expected = cudf::table_view{{lists_col->view()}};
  run_test(expected);
}

TEST_F(CutableTest, StructsContainingLists)
{
  cudf::test::lists_column_wrapper<int32_t> list_col{{1, 2, 3}, {4, 5}, {}, {6}};
  cudf::test::fixed_width_column_wrapper<int32_t> int_col{10, 20, 30, 40};
  cudf::test::structs_column_wrapper struct_col{{list_col, int_col}};

  auto const expected = cudf::table_view{{struct_col}};
  run_test(expected);
}

TEST_F(CutableTest, NestedStructs)
{
  cudf::test::fixed_width_column_wrapper<int32_t> inner_col1{1, 2};
  cudf::test::strings_column_wrapper inner_col2{"a", "b"};
  cudf::test::structs_column_wrapper inner_struct{{inner_col1, inner_col2}};

  cudf::test::fixed_width_column_wrapper<float> outer_col{1.5f, 2.5f};
  cudf::test::structs_column_wrapper outer_struct{{inner_struct, outer_col}};

  auto const expected = cudf::table_view{{outer_struct}};
  run_test(expected);
}

// ============================================================================
// I/O type tests
// ============================================================================

TEST_F(CutableTest, DeviceBufferSource)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});

  auto const expected = cudf::table_view{{col}};
  run_test_device_buffer(expected);
}

TEST_F(CutableTest, VoidSink)
{
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5});
  auto const expected = cudf::table_view{{col}};

  // Write to void sink (should succeed but data is discarded)
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{}, expected).build());

  // Test passes if write doesn't throw
  EXPECT_TRUE(true);
}

// ============================================================================
// Additional edge cases
// ============================================================================

TEST_F(CutableTest, UnicodeStrings)
{
  cudf::test::strings_column_wrapper col({"Hello", "世界", "🌍", "café", "résumé"},
                                         {true, true, true, false, true});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, VeryLongStrings)
{
  std::string long_str(10000, 'a');
  cudf::test::strings_column_wrapper col({long_str, "short", long_str + "suffix"});

  auto const expected = cudf::table_view{{col}};
  run_test(expected);
}

TEST_F(CutableTest, ManyColumns)
{
  constexpr int num_cols = 50;
  std::vector<cudf::column_view> columns;
  for (int i = 0; i < num_cols; ++i) {
    cudf::test::fixed_width_column_wrapper<int32_t> col({i, i + 1, i + 2});
    columns.push_back(col);
  }

  cudf::table_view expected(columns);
  run_test(expected);
}

TEST_F(CutableTest, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int32_t> empty_col({});
  cudf::test::fixed_width_column_wrapper<int32_t> non_empty_col({1, 2, 3});

  auto const expected = cudf::table_view{{empty_col, non_empty_col}};
  run_test(expected);
}

// ============================================================================
// Additional error handling tests
// ============================================================================

TEST_F(CutableTest, FileTooSmallForHeader)
{
  auto const filepath = temp_env->get_temp_filepath("too_small.cutable");

  // Create a file that's too small to contain a header
  std::ofstream file(filepath, std::ios::binary);
  file.write("SMALL", 5);
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

TEST_F(CutableTest, CorruptedMetadataLength)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_meta.cutable");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

  // Get the actual file size
  std::ifstream size_check(filepath, std::ios::binary | std::ios::ate);
  auto const file_size = size_check.tellg();
  size_check.close();

  // Corrupt the metadata length (set to a value larger than the file)
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(uint32_t) * 2);                                    // Skip magic and version
  uint64_t bad_meta_length = static_cast<uint64_t>(file_size) + 1000;  // Larger than file
  file.write(reinterpret_cast<char*>(&bad_meta_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

TEST_F(CutableTest, CorruptedDataLength)
{
  auto const filepath = temp_env->get_temp_filepath("corrupted_data.cutable");
  cudf::test::fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto const expected = cudf::table_view{{col}};
  cudf::io::experimental::write_cutable(
    cudf::io::cutable_writer_options::builder(cudf::io::sink_info{filepath}, expected).build());

  // Get the actual file size
  std::ifstream size_check(filepath, std::ios::binary | std::ios::ate);
  auto const file_size = size_check.tellg();
  size_check.close();

  // Read the header to get the actual metadata_length
  std::ifstream header_read(filepath, std::ios::binary);
  uint32_t magic, version;
  uint64_t metadata_length;
  header_read.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
  header_read.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
  header_read.read(reinterpret_cast<char*>(&metadata_length), sizeof(uint64_t));
  header_read.close();

  // Corrupt the data length (set to a value that makes total size exceed file size)
  std::fstream file(filepath, std::ios::in | std::ios::out | std::ios::binary);
  file.seekp(sizeof(uint32_t) * 2 + sizeof(uint64_t));  // Skip magic, version, and metadata_length
  // Set data_length so that metadata_offset + metadata_length + data_length > file_size
  size_t header_size = sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2;  // magic + version + 2 lengths
  size_t metadata_offset = header_size;
  size_t min_data_length_needed =
    static_cast<size_t>(file_size) - metadata_offset - metadata_length + 1;
  uint64_t bad_data_length = min_data_length_needed;
  file.write(reinterpret_cast<char*>(&bad_data_length), sizeof(uint64_t));
  file.close();

  EXPECT_THROW(
    cudf::io::experimental::read_cutable(
      cudf::io::cutable_reader_options::builder(cudf::io::source_info{filepath}).build()),
    cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
