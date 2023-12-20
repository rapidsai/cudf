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

#include "parquet_common.hpp"

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <src/io/parquet/compact_protocol_reader.hpp>
#include <src/io/parquet/parquet.hpp>
#include <src/io/parquet/parquet_gpu.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <fstream>

// Global environment for temporary files
cudf::test::TempDirTestEnvironment* const temp_env =
  static_cast<cudf::test::TempDirTestEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

// Declare typed test cases
TYPED_TEST_SUITE(ParquetWriterNumericTypeTest, SupportedTypes);
TYPED_TEST_SUITE(ParquetWriterComparableTypeTest, ComparableAndFixedTypes);
TYPED_TEST_SUITE(ParquetWriterChronoTypeTest, cudf::test::ChronoTypes);
TYPED_TEST_SUITE(ParquetWriterTimestampTypeTest, SupportedTimestampTypes);
TYPED_TEST_SUITE(ParquetWriterSchemaTest, cudf::test::AllTypes);
TYPED_TEST_SUITE(ParquetReaderSourceTest, ByteLikeTypes);

// Declare typed test cases
TYPED_TEST_SUITE(ParquetChunkedWriterNumericTypeTest, SupportedTypes);

INSTANTIATE_TEST_SUITE_P(ParquetV2ReadWriteTest,
                         ParquetV2Test,
                         testing::Bool(),
                         testing::PrintToStringParamName());

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i % 400); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 800;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumn.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumnWithNulls)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("SingleColumnWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, Timestamps)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("Timestamps.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampsWithNulls)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });
  auto validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i > 30) && (i < 60); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);

  auto expected = table_view{{col}};

  auto filepath = temp_env->get_temp_filepath("TimestampsWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetWriterTimestampTypeTest, TimestampOverflow)
{
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  auto sequence = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return max - i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return true; });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, validity);
  table_view expected({col});

  auto filepath = temp_env->get_temp_filepath("ParquetTimestampOverflow.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .timestamp_type(this->type());
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize)
{
  // write out two 31 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 31;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,

                 true,  true, true, true, true, true, true, true, true};
  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TYPED_TEST(ParquetChunkedWriterNumericTypeTest, UnalignedSize2)
{
  // write out two 33 row tables and make sure they get
  // read back with all their validity bits in the right place

  using T = TypeParam;

  int num_els = 33;
  std::vector<std::unique_ptr<cudf::column>> cols;

  bool mask[] = {false, true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true,
                 true,  true, true, true, true, true, true, true, true, true, true};

  T c1a[num_els];
  std::fill(c1a, c1a + num_els, static_cast<T>(5));
  T c1b[num_els];
  std::fill(c1b, c1b + num_els, static_cast<T>(6));
  column_wrapper<T> c1a_w(c1a, c1a + num_els, mask);
  column_wrapper<T> c1b_w(c1b, c1b + num_els, mask);
  cols.push_back(c1a_w.release());
  cols.push_back(c1b_w.release());
  cudf::table tbl1(std::move(cols));

  T c2a[num_els];
  std::fill(c2a, c2a + num_els, static_cast<T>(8));
  T c2b[num_els];
  std::fill(c2b, c2b + num_els, static_cast<T>(9));
  column_wrapper<T> c2a_w(c2a, c2a + num_els, mask);
  column_wrapper<T> c2b_w(c2b, c2b + num_els, mask);
  cols.push_back(c2a_w.release());
  cols.push_back(c2b_w.release());
  cudf::table tbl2(std::move(cols));

  auto expected = cudf::concatenate(std::vector<table_view>({tbl1, tbl2}));

  auto filepath = temp_env->get_temp_filepath("ChunkedUnalignedSize2.parquet");
  cudf::io::chunked_parquet_writer_options args =
    cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info{filepath});
  cudf::io::parquet_chunked_writer(args).write(tbl1).write(tbl2);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

// custom mem mapped data sink that supports device writes
template <bool supports_device_writes>
class custom_test_memmap_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_memmap_sink(std::vector<char>* mm_writer_buf)
  {
    mm_writer = cudf::io::data_sink::create(mm_writer_buf);
  }

  virtual ~custom_test_memmap_sink() { mm_writer->flush(); }

  void host_write(void const* data, size_t size) override { mm_writer->host_write(data, size); }

  [[nodiscard]] bool supports_device_write() const override { return supports_device_writes; }

  void device_write(void const* gpu_data, size_t size, rmm::cuda_stream_view stream) override
  {
    this->device_write_async(gpu_data, size, stream).get();
  }

  std::future<void> device_write_async(void const* gpu_data,
                                       size_t size,
                                       rmm::cuda_stream_view stream) override
  {
    return std::async(std::launch::deferred, [=] {
      char* ptr = nullptr;
      CUDF_CUDA_TRY(cudaMallocHost(&ptr, size));
      CUDF_CUDA_TRY(cudaMemcpyAsync(ptr, gpu_data, size, cudaMemcpyDefault, stream.value()));
      stream.synchronize();
      mm_writer->host_write(ptr, size);
      CUDF_CUDA_TRY(cudaFreeHost(ptr));
    });
  }

  void flush() override { mm_writer->flush(); }

  size_t bytes_written() override { return mm_writer->bytes_written(); }

 private:
  std::unique_ptr<data_sink> mm_writer;
};

TEST_F(ParquetWriterStressTest, LargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, LargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<false> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWeakCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(16, 4 * 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableGoodCompression)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 128 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterStressTest, DeviceWriteLargeTableWithValids)
{
  std::vector<char> mm_buf;
  mm_buf.reserve(4 * 1024 * 1024 * 16);
  custom_test_memmap_sink<true> custom_sink(&mm_buf);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 6, true);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{mm_buf.data(), mm_buf.size()});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TYPED_TEST(ParquetWriterComparableTypeTest, ThreeColumnSorted)
{
  using T = TypeParam;

  auto col0 = testdata::ascending<T>();
  auto col1 = testdata::descending<T>();
  auto col2 = testdata::unordered<T>();

  auto const expected = table_view{{col0, col1, col2}};

  auto const filepath = temp_env->get_temp_filepath("ThreeColumnSorted.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_size_for_ordered_tests)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);

  auto const& columns = fmd.row_groups[0].columns;
  ASSERT_EQ(columns.size(), static_cast<size_t>(expected.num_columns()));

  // now check that the boundary order for chunk 1 is ascending,
  // chunk 2 is descending, and chunk 3 is unordered
  cudf::io::parquet::detail::BoundaryOrder expected_orders[] = {
    cudf::io::parquet::detail::BoundaryOrder::ASCENDING,
    cudf::io::parquet::detail::BoundaryOrder::DESCENDING,
    cudf::io::parquet::detail::BoundaryOrder::UNORDERED};

  for (std::size_t i = 0; i < columns.size(); i++) {
    auto const ci = read_column_index(source, columns[i]);
    EXPECT_EQ(ci.boundary_order, expected_orders[i]);
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(
        cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }

  {
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(cudf::host_span<T const>(
        reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*table, result.tbl->view());
  }
}

TYPED_TEST(ParquetReaderSourceTest, BufferSourceArrayTypes)
{
  using T = TypeParam;

  srand(31337);
  auto table = create_random_fixed_table<int>(5, 5, true);

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), *table);
  cudf::io::write_parquet(out_opts);

  auto full_table = cudf::concatenate(std::vector<table_view>({*table, *table}));

  {
    auto spans = std::vector<cudf::host_span<T>>{
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T>(reinterpret_cast<T*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }

  {
    auto spans = std::vector<cudf::host_span<T const>>{
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size()),
      cudf::host_span<T const>(reinterpret_cast<T const*>(out_buffer.data()), out_buffer.size())};
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(cudf::host_span<cudf::host_span<T const>>(spans.data(), spans.size())));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(*full_table, result.tbl->view());
  }
}

TEST_F(ParquetMetadataReaderTest, TestBasic)
{
  auto const num_rows = 1200;

  auto ints   = random_values<int>(num_rows);
  auto floats = random_values<float>(num_rows);
  column_wrapper<int> int_col(ints.begin(), ints.end());
  column_wrapper<float> float_col(floats.begin(), floats.end());

  table_view expected({int_col, float_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int_col");
  expected_metadata.column_metadata[1].set_name("float_col");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 int_col
 float_col
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  EXPECT_EQ(meta.schema().root().child(0).name(), "int_col");
  EXPECT_EQ(meta.schema().root().child(1).name(), "float_col");
}

TEST_F(ParquetMetadataReaderTest, TestNested)
{
  auto const num_rows       = 1200;
  auto const lists_per_row  = 4;
  auto const num_child_rows = num_rows * lists_per_row;

  auto keys = random_values<int>(num_child_rows);
  auto vals = random_values<float>(num_child_rows);
  column_wrapper<int> keys_col(keys.begin(), keys.end());
  column_wrapper<float> vals_col(vals.begin(), vals.end());
  auto s_col = cudf::test::structs_column_wrapper({keys_col, vals_col}).release();

  std::vector<int> row_offsets(num_rows + 1);
  for (int idx = 0; idx < num_rows + 1; ++idx) {
    row_offsets[idx] = idx * lists_per_row;
  }
  column_wrapper<int> offsets(row_offsets.begin(), row_offsets.end());

  auto list_col =
    cudf::make_lists_column(num_rows, offsets.release(), std::move(s_col), 0, rmm::device_buffer{});

  table_view expected({*list_col, *list_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("maps");
  expected_metadata.column_metadata[0].set_list_column_as_map();
  expected_metadata.column_metadata[1].set_name("lists");
  expected_metadata.column_metadata[1].child(1).child(0).set_name("int_field");
  expected_metadata.column_metadata[1].child(1).child(1).set_name("float_field");

  auto filepath = temp_env->get_temp_filepath("MetadataTest.orc");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(out_opts);

  auto meta = read_parquet_metadata(cudf::io::source_info{filepath});
  EXPECT_EQ(meta.num_rows(), num_rows);

  std::string expected_schema = R"(schema
 maps
  key_value
   key
   value
 lists
  list
   element
    int_field
    float_field
)";
  EXPECT_EQ(expected_schema, print(meta.schema().root()));

  EXPECT_EQ(meta.schema().root().name(), "schema");
  EXPECT_EQ(meta.schema().root().type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(meta.schema().root().num_children(), 2);

  auto const& out_map_col = meta.schema().root().child(0);
  EXPECT_EQ(out_map_col.name(), "maps");
  EXPECT_EQ(out_map_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // map

  ASSERT_EQ(out_map_col.num_children(), 1);
  EXPECT_EQ(out_map_col.child(0).name(), "key_value");  // key_value (named in parquet writer)
  ASSERT_EQ(out_map_col.child(0).num_children(), 2);
  EXPECT_EQ(out_map_col.child(0).child(0).name(), "key");    // key (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(1).name(), "value");  // value (named in parquet writer)
  EXPECT_EQ(out_map_col.child(0).child(0).type_kind(), cudf::io::parquet::TypeKind::INT32);  // int
  EXPECT_EQ(out_map_col.child(0).child(1).type_kind(),
            cudf::io::parquet::TypeKind::FLOAT);  // float

  auto const& out_list_col = meta.schema().root().child(1);
  EXPECT_EQ(out_list_col.name(), "lists");
  EXPECT_EQ(out_list_col.type_kind(), cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // list
  // TODO repetition type?
  ASSERT_EQ(out_list_col.num_children(), 1);
  EXPECT_EQ(out_list_col.child(0).name(), "list");  // list (named in parquet writer)
  ASSERT_EQ(out_list_col.child(0).num_children(), 1);

  auto const& out_list_struct_col = out_list_col.child(0).child(0);
  EXPECT_EQ(out_list_struct_col.name(), "element");  // elements (named in parquet writer)
  EXPECT_EQ(out_list_struct_col.type_kind(),
            cudf::io::parquet::TypeKind::UNDEFINED_TYPE);  // struct
  ASSERT_EQ(out_list_struct_col.num_children(), 2);

  auto const& out_int_col = out_list_struct_col.child(0);
  EXPECT_EQ(out_int_col.name(), "int_field");
  EXPECT_EQ(out_int_col.type_kind(), cudf::io::parquet::TypeKind::INT32);

  auto const& out_float_col = out_list_struct_col.child(1);
  EXPECT_EQ(out_float_col.name(), "float_field");
  EXPECT_EQ(out_float_col.type_kind(), cudf::io::parquet::TypeKind::FLOAT);
}

TYPED_TEST_SUITE(ParquetReaderPredicatePushdownTest, SupportedTestTypes);

TYPED_TEST(ParquetReaderPredicatePushdownTest, FilterTyped)
{
  using T = TypeParam;

  auto const [src, filepath] = create_parquet_typed_with_stats<T>("FilterTyped.parquet");
  auto const written_table   = src.view();

  // Filtering AST
  auto literal_value = []() {
    if constexpr (cudf::is_timestamp<T>()) {
      // table[0] < 10000 timestamp days/seconds/milliseconds/microseconds/nanoseconds
      return cudf::timestamp_scalar<T>(T(typename T::duration(10000)));  // i (0-20,000)
    } else if constexpr (cudf::is_duration<T>()) {
      // table[0] < 10000 day/seconds/milliseconds/microseconds/nanoseconds
      return cudf::duration_scalar<T>(T(10000));  // i (0-20,000)
    } else if constexpr (std::is_same_v<T, cudf::string_view>) {
      // table[0] < "000010000"
      return cudf::string_scalar("000010000");  // i (0-20,000)
    } else {
      // table[0] < 0 or 100u
      return cudf::numeric_scalar<T>((100 - 100 * std::is_signed_v<T>));  // i/100 (-100-100/ 0-200)
    }
  }();
  auto literal           = cudf::ast::literal(literal_value);
  auto col_name_0        = cudf::ast::column_name_reference("col0");
  auto filter_expression = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_name_0, literal);
  auto col_ref_0         = cudf::ast::column_reference(0);
  auto ref_filter        = cudf::ast::operation(cudf::ast::ast_operator::LESS, col_ref_0, literal);

  // Expected result
  auto predicate = cudf::compute_column(written_table, ref_filter);
  EXPECT_EQ(predicate->view().type().id(), cudf::type_id::BOOL8)
    << "Predicate filter should return a boolean";
  auto expected = cudf::apply_boolean_mask(written_table, *predicate);

  // Reading with Predicate Pushdown
  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .filter(filter_expression);
  auto result       = cudf::io::read_parquet(read_opts);
  auto result_table = result.tbl->view();

  // tests
  EXPECT_EQ(int(written_table.column(0).type().id()), int(result_table.column(0).type().id()))
    << "col0 type mismatch";
  // To make sure AST filters out some elements
  EXPECT_LT(expected->num_rows(), written_table.num_rows());
  EXPECT_EQ(result_table.num_rows(), expected->num_rows());
  EXPECT_EQ(result_table.num_columns(), expected->num_columns());
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected->view(), result_table);
}

CUDF_TEST_PROGRAM_MAIN()
