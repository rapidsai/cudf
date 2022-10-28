/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/span.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <rmm/cuda_stream_view.hpp>

#include <fstream>
#include <type_traits>

namespace {
// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

using int32s_col       = cudf::test::fixed_width_column_wrapper<int32_t>;
using int64s_col       = cudf::test::fixed_width_column_wrapper<int64_t>;
using strings_col      = cudf::test::strings_column_wrapper;
using structs_col      = cudf::test::structs_column_wrapper;
using int32s_lists_col = cudf::test::lists_column_wrapper<int32_t>;

auto write_file(std::vector<std::unique_ptr<cudf::column>>& input_columns,
                std::string const& filename,
                bool nullable,
                std::size_t max_page_size_bytes = cudf::io::default_max_page_size_bytes,
                std::size_t max_page_size_rows  = cudf::io::default_max_page_size_rows)
{
  // Just shift nulls of the next column by one position to avoid having all nulls in the same
  // table rows.
  if (nullable) {
    // Generate deterministic bitmask instead of random bitmask for easy computation of data size.
    auto const valid_iter = cudf::detail::make_counting_transform_iterator(
      0, [](cudf::size_type i) { return i % 4 == 3 ? 0 : 1; });

    cudf::size_type offset{0};
    for (auto& col : input_columns) {
      col->set_null_mask(
        cudf::test::detail::make_null_mask(valid_iter + offset, valid_iter + col->size() + offset));

      if (col->type().id() == cudf::type_id::STRUCT) {
        auto const null_mask  = col->view().null_mask();
        auto const null_count = col->null_count();
        for (cudf::size_type idx = 0; idx < col->num_children(); ++idx) {
          cudf::structs::detail::superimpose_parent_nulls(null_mask,
                                                          null_count,
                                                          col->child(idx),
                                                          cudf::get_default_stream(),
                                                          rmm::mr::get_current_device_resource());
        }
      } else if (col->type().id() == cudf::type_id::LIST) {
        col = cudf::purge_nonempty_nulls(cudf::lists_column_view{col->view()});
      }
    }
  }
  auto input_table = std::make_unique<cudf::table>(std::move(input_columns));
  auto filepath =
    temp_env->get_temp_filepath(nullable ? filename + "_nullable.parquet" : filename + ".parquet");

  auto const write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *input_table)
      .max_page_size_bytes(max_page_size_bytes)
      .max_page_size_rows(max_page_size_rows)
      .build();
  cudf::io::write_parquet(write_opts);

  return std::pair{std::move(input_table), std::move(filepath)};
}

auto chunked_read(std::string const& filepath, std::size_t byte_limit)
{
  auto const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_parquet_reader(byte_limit, read_opts);

  auto num_chunks = 0;
  auto result     = std::make_unique<cudf::table>();

  do {
    auto chunk = reader.read_chunk();
    if (num_chunks == 0) {
      result = std::move(chunk.tbl);
    } else {
      CUDF_EXPECTS(chunk.tbl->num_rows() != 0, "Number of rows in the new chunk is zero.");
      result = cudf::concatenate(std::vector<cudf::table_view>{result->view(), chunk.tbl->view()});
    }
    ++num_chunks;

    if (result->num_rows() == 0) { break; }
  } while (reader.has_next());

  return std::pair(std::move(result), num_chunks);
}

}  // namespace

struct ParquetChunkedReaderTest : public cudf::test::BaseFixture {
};

TEST_F(ParquetChunkedReaderTest, TestChunkedReadNoData)
{
  auto const do_test = []() {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    input_columns.emplace_back(int32s_col{}.release());
    input_columns.emplace_back(int64s_col{}.release());

    auto [input_table, filepath] = write_file(input_columns, "chunked_read_empty", false);
    auto [result, num_chunks]    = chunked_read(filepath, 1'000);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  auto const [expected, result, num_chunks] = do_test();
  EXPECT_EQ(num_chunks, 1);
  EXPECT_EQ(result->num_rows(), 0);
  EXPECT_EQ(result->num_columns(), 2);
  CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadSimpleData)
{
  auto constexpr num_rows = 40'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit, bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());
    input_columns.emplace_back(int64s_col(value_iter, value_iter + num_rows).release());

    auto [input_table, filepath] = write_file(input_columns, "chunked_read_simple", nullable);
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  {
    auto const [expected, result, num_chunks] = do_test(240'000, false);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, result, num_chunks] = do_test(240'000, true);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadBoundaryCases)
{
  // tests some specific boundary conditions in the split calculations.

  auto constexpr num_rows = 40'000;

  auto const do_test = [](std::size_t chunk_read_limit) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());

    auto [input_table, filepath] = write_file(input_columns, "chunked_read_simple_boundary", false);
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  // test with a limit slightly less than one page of data
  {
    auto const [expected, result, num_chunks] = do_test(79'000);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit exactly the size one page of data
  {
    auto const [expected, result, num_chunks] = do_test(80'000);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit slightly more the size one page of data
  {
    auto const [expected, result, num_chunks] = do_test(81'000);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit slightly less than two pages of data
  {
    auto const [expected, result, num_chunks] = do_test(159'000);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit exactly the size of two pages of data minus one byte
  {
    auto const [expected, result, num_chunks] = do_test(159'999);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit exactly the size of two pages of data
  {
    auto const [expected, result, num_chunks] = do_test(160'000);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // test with a limit slightly more the size two pages of data
  {
    auto const [expected, result, num_chunks] = do_test(161'000);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithString)
{
  auto constexpr num_rows = 60'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit, bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const value_iter = thrust::make_counting_iterator(0);

    // ints                                            Page    total bytes   cumulative bytes
    // 20000 rows of 4 bytes each                    = A0      80000         80000
    // 20000 rows of 4 bytes each                    = A1      80000         160000
    // 20000 rows of 4 bytes each                    = A2      80000         240000
    input_columns.emplace_back(int32s_col(value_iter, value_iter + num_rows).release());

    // strings                                         Page    total bytes   cumulative bytes
    // 20000 rows of 1 char each    (20000  + 80004) = B0      100004        100004
    // 20000 rows of 4 chars each   (80000  + 80004) = B1      160004        260008
    // 20000 rows of 16 chars each  (320000 + 80004) = B2      400004        660012
    auto const strings  = std::vector<std::string>{"a", "bbbb", "cccccccccccccccc"};
    auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
      if (i < 20000) { return strings[0]; }
      if (i < 40000) { return strings[1]; }
      return strings[2];
    });
    input_columns.emplace_back(strings_col(str_iter, str_iter + num_rows).release());

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_strings",
                                              nullable,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );

    // Cumulative sizes:
    // A0 + B0 :  180004
    // A1 + B1 :  420008
    // A2 + B2 :  900012
    //                                    skip_rows / num_rows
    // byte_limit==500000  should give 2 chunks: {0, 40000}, {40000, 20000}
    // byte_limit==1000000 should give 1 chunks: {0, 60000},
    auto [result, num_chunks] = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  {
    auto const [expected, result, num_chunks] = do_test(500'000, false);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, result, num_chunks] = do_test(500'000, true);
    EXPECT_EQ(num_chunks, 2);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, result, num_chunks] = do_test(1'000'000, false);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
  {
    auto const [expected, result, num_chunks] = do_test(1'000'000, true);
    EXPECT_EQ(num_chunks, 1);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithStructs)
{
  auto constexpr num_rows = 100'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit, bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      auto child1 = int32s_col(int_iter, int_iter + num_rows);
      auto child2 = int32s_col(int_iter + num_rows, int_iter + num_rows * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_rows};

      return structs_col{{child1, child2, child3}}.release();
    }());

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_structs",
                                              nullable,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  {
    auto const [expected, result, num_chunks] = do_test(500'000, false);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, result, num_chunks] = do_test(500'000, true);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithListsNoNulls)
{
  auto constexpr num_rows = 100'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 page consist of:
    //
    // 20001 offsets :   80004  bytes
    // 30000 ints    :   120000 bytes
    // total         :   200004 bytes
    auto const template_lists = int32s_lists_col{
      int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{1, 2}, int32s_lists_col{3, 4, 5}};

    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_lists",
                                              false /*nullable*/,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  // chunk size slightly less than 1 page (forcing it to be at least 1 page per read)
  {
    auto const [expected, result, num_chunks] = do_test(200'000);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // chunk size exactly 1 page
  {
    auto const [expected, result, num_chunks] = do_test(200'004);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // chunk size 2 pages. 3 chunks (2 pages + 2 pages + 1 page)
  {
    auto const [expected, result, num_chunks] = do_test(400'008);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // chunk size 2 pages minus one byte: each chunk will be just one page
  {
    auto const [expected, result, num_chunks] = do_test(400'007);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithListsHavingNulls)
{
  auto constexpr num_rows = 100'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    // 20000 rows in 1 page consist of:
    //
    // 625 validity words :   2500 bytes   (a null every 4 rows: null at indices [3, 7, 11, ...])
    // 20001 offsets      :   80004  bytes
    // 15000 ints         :   60000 bytes
    // total              :   142504 bytes
    auto const template_lists =
      int32s_lists_col{// these will all be null
                       int32s_lists_col{},
                       int32s_lists_col{0},
                       int32s_lists_col{1, 2},
                       int32s_lists_col{3, 4, 5, 6, 7, 8, 9} /* this list will be nullified out */};
    auto const gather_iter =
      cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
    auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
    input_columns.emplace_back(
      std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_lists_nulls",
                                              true /*nullable*/,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  // chunk size slightly less than 1 page (forcing it to be at least 1 page per read)
  {
    auto const [input, result, num_chunks] = do_test(142'500);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*input, *result);
  }

  // chunk size exactly 1 page
  {
    auto const [input, result, num_chunks] = do_test(142'504);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*input, *result);
  }

  // chunk size 2 pages. 3 chunks (2 pages + 2 pages + 1 page)
  {
    auto const [expected, result, num_chunks] = do_test(285'008);
    EXPECT_EQ(num_chunks, 3);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  // chunk size 2 pages minus 1 byte: each chunk will be just one page
  {
    auto const [expected, result, num_chunks] = do_test(285'007);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithStructsOfLists)
{
  auto constexpr num_rows = 100'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit, bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
    input_columns.emplace_back([=] {
      std::vector<std::unique_ptr<cudf::column>> child_columns;
      child_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());
      child_columns.emplace_back(
        int32s_col(int_iter + num_rows, int_iter + num_rows * 2).release());

      auto const str_iter = cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) {
        return std::to_string(i) + "++++++++++++++++++++" + std::to_string(i);
      });
      child_columns.emplace_back(strings_col{str_iter, str_iter + num_rows}.release());

      auto const template_lists = int32s_lists_col{
        int32s_lists_col{}, int32s_lists_col{0}, int32s_lists_col{0, 1}, int32s_lists_col{0, 1, 2}};
      auto const gather_iter =
        cudf::detail::make_counting_transform_iterator(0, [&](int32_t i) { return i % 4; });
      auto const gather_map = int32s_col(gather_iter, gather_iter + num_rows);
      child_columns.emplace_back(
        std::move(cudf::gather(cudf::table_view{{template_lists}}, gather_map)->release().front()));

      return structs_col(std::move(child_columns)).release();
    }());

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_structs_of_lists",
                                              nullable,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  {
    auto const [expected, result, num_chunks] = do_test(500'000, false);
    EXPECT_EQ(num_chunks, 10);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, result, num_chunks] = do_test(500'000, true);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}

TEST_F(ParquetChunkedReaderTest, TestChunkedReadWithListsOfStructs)
{
  auto constexpr num_rows = 100'000;

  auto const do_test = [num_rows](std::size_t chunk_read_limit, bool nullable) {
    std::vector<std::unique_ptr<cudf::column>> input_columns;
    auto const int_iter = thrust::make_counting_iterator(0);
    input_columns.emplace_back(int32s_col(int_iter, int_iter + num_rows).release());

    auto offsets = std::vector<cudf::size_type>{};
    offsets.reserve(num_rows * 2);
    cudf::size_type num_structs = 0;
    for (int i = 0; i < num_rows; ++i) {
      offsets.push_back(num_structs);
      auto const new_list_size = i % 4;
      num_structs += new_list_size;
    }
    offsets.push_back(num_structs);

    auto const make_structs_col = [=] {
      auto child1 = int32s_col(int_iter, int_iter + num_structs);
      auto child2 = int32s_col(int_iter + num_structs, int_iter + num_structs * 2);

      auto const str_iter = cudf::detail::make_counting_transform_iterator(
        0, [&](int32_t i) { return std::to_string(i) + std::to_string(i) + std::to_string(i); });
      auto child3 = strings_col{str_iter, str_iter + num_structs};

      return structs_col{{child1, child2, child3}}.release();
    };

    input_columns.emplace_back(
      cudf::make_lists_column(static_cast<cudf::size_type>(offsets.size() - 1),
                              int32s_col(offsets.begin(), offsets.end()).release(),
                              make_structs_col(),
                              0,
                              rmm::device_buffer{}));

    auto [input_table, filepath] = write_file(input_columns,
                                              "chunked_read_with_lists_of_structs",
                                              nullable,
                                              512 * 1024,  // 512KB per page
                                              20000        // 20k rows per page
    );
    auto [result, num_chunks]    = chunked_read(filepath, chunk_read_limit);
    return std::tuple{std::move(input_table), std::move(result), num_chunks};
  };

  {
    auto const [expected, result, num_chunks] = do_test(1'000'000, false);
    EXPECT_EQ(num_chunks, 7);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }

  {
    auto const [expected, result, num_chunks] = do_test(1'000'000, true);
    EXPECT_EQ(num_chunks, 5);
    CUDF_TEST_EXPECT_TABLES_EQUAL(*expected, *result);
  }
}
