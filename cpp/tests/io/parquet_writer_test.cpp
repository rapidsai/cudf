/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/io_metadata_utilities.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>

#include <cudf/io/data_sink.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/unary.hpp>

#include <fstream>

using cudf::test::iterators::no_nulls;

template <typename mask_op_t>
void test_durations(mask_op_t mask_op)
{
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution_d(0, 30);
  auto sequence_d = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution_d(generator); });

  std::uniform_int_distribution<int> distribution_s(0, 86400);
  auto sequence_s = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution_s(generator); });

  std::uniform_int_distribution<int> distribution(0, 86400 * 1000);
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [&](auto i) { return distribution(generator); });

  auto mask = cudf::detail::make_counting_transform_iterator(0, mask_op);

  constexpr auto num_rows = 100;
  // Durations longer than a day are not exactly valid, but cudf should be able to round trip
  auto durations_d = cudf::test::fixed_width_column_wrapper<cudf::duration_D, int64_t>(
    sequence_d, sequence_d + num_rows, mask);
  auto durations_s = cudf::test::fixed_width_column_wrapper<cudf::duration_s, int64_t>(
    sequence_s, sequence_s + num_rows, mask);
  auto durations_ms = cudf::test::fixed_width_column_wrapper<cudf::duration_ms, int64_t>(
    sequence, sequence + num_rows, mask);
  auto durations_us = cudf::test::fixed_width_column_wrapper<cudf::duration_us, int64_t>(
    sequence, sequence + num_rows, mask);
  auto durations_ns = cudf::test::fixed_width_column_wrapper<cudf::duration_ns, int64_t>(
    sequence, sequence + num_rows, mask);

  auto expected = table_view{{durations_d, durations_s, durations_ms, durations_us, durations_ns}};

  auto filepath = temp_env->get_temp_filepath("Durations.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);

  auto durations_d_got =
    cudf::cast(result.tbl->view().column(0), cudf::data_type{cudf::type_id::DURATION_DAYS});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_d, durations_d_got->view());

  auto durations_s_got =
    cudf::cast(result.tbl->view().column(1), cudf::data_type{cudf::type_id::DURATION_SECONDS});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_s, durations_s_got->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_ms, result.tbl->view().column(2));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_us, result.tbl->view().column(3));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(durations_ns, result.tbl->view().column(4));
}

TEST_F(ParquetWriterTest, Durations)
{
  test_durations([](auto i) { return true; });
  test_durations([](auto i) { return (i % 2) != 0; });
  test_durations([](auto i) { return (i % 3) != 0; });
  test_durations([](auto i) { return false; });
}

TEST_F(ParquetWriterTest, MultiIndex)
{
  constexpr auto num_rows = 100;

  auto col0_data = random_values<int8_t>(num_rows);
  auto col1_data = random_values<int16_t>(num_rows);
  auto col2_data = random_values<int32_t>(num_rows);
  auto col3_data = random_values<float>(num_rows);
  auto col4_data = random_values<double>(num_rows);

  column_wrapper<int8_t> col0{col0_data.begin(), col0_data.end(), no_nulls()};
  column_wrapper<int16_t> col1{col1_data.begin(), col1_data.end(), no_nulls()};
  column_wrapper<int32_t> col2{col2_data.begin(), col2_data.end(), no_nulls()};
  column_wrapper<float> col3{col3_data.begin(), col3_data.end(), no_nulls()};
  column_wrapper<double> col4{col4_data.begin(), col4_data.end(), no_nulls()};

  auto expected = table_view{{col0, col1, col2, col3, col4}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("int8s");
  expected_metadata.column_metadata[1].set_name("int16s");
  expected_metadata.column_metadata[2].set_name("int32s");
  expected_metadata.column_metadata[3].set_name("floats");
  expected_metadata.column_metadata[4].set_name("doubles");

  auto filepath = temp_env->get_temp_filepath("MultiIndex.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata)
      .key_value_metadata(
        {{{"pandas", "\"index_columns\": [\"int8s\", \"int16s\"], \"column1\": [\"int32s\"]"}}});
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .use_pandas_metadata(true)
      .columns({"int32s", "floats", "doubles"});
  auto result = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
  cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
}

TEST_F(ParquetWriterTest, BufferSource)
{
  constexpr auto num_rows = 100 << 10;
  auto const seq_col      = random_values<int>(num_rows);
  column_wrapper<int> col{seq_col.begin(), seq_col.end(), no_nulls()};

  auto const expected = table_view{{col}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("col_other");

  std::vector<char> out_buffer;
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer), expected)
      .metadata(expected_metadata);
  cudf::io::write_parquet(out_opts);

  // host buffer
  {
    cudf::io::parquet_reader_options in_opts = cudf::io::parquet_reader_options::builder(
      cudf::io::source_info(out_buffer.data(), out_buffer.size()));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }

  // device buffer
  {
    auto const d_input = cudf::detail::make_device_uvector_sync(
      cudf::host_span<uint8_t const>{reinterpret_cast<uint8_t const*>(out_buffer.data()),
                                     out_buffer.size()},
      cudf::get_default_stream(),
      rmm::mr::get_current_device_resource());
    auto const d_buffer = cudf::device_span<std::byte const>(
      reinterpret_cast<std::byte const*>(d_input.data()), d_input.size());
    cudf::io::parquet_reader_options in_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info(d_buffer));
    auto const result = cudf::io::read_parquet(in_opts);

    CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
    cudf::test::expect_metadata_equal(expected_metadata, result.metadata);
  }
}

TEST_F(ParquetWriterTest, ManyFragments)
{
  srand(31337);
  auto const expected = create_random_fixed_table<int>(1, 700'000, false);

  auto const filepath = temp_env->get_temp_filepath("ManyFragments.parquet");
  cudf::io::parquet_writer_options const args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected)
      .max_page_size_bytes(8 * 1024)
      .max_page_fragment_size(10);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options const read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetWriterTest, NonNullable)
{
  srand(31337);
  auto expected = create_random_fixed_table<int>(9, 9, false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, *expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, *expected);
}

TEST_F(ParquetWriterTest, Struct)
{
  // Struct<is_human:bool, Struct<names:string, ages:int>>

  auto names = {"Samuel Vimes",
                "Carrot Ironfoundersson",
                "Angua von Uberwald",
                "Cheery Littlebottom",
                "Detritus",
                "Mr Slant"};

  // `Name` column has all valid values.
  auto names_col = cudf::test::strings_column_wrapper{names.begin(), names.end()};

  auto ages_col =
    cudf::test::fixed_width_column_wrapper<int32_t>{{48, 27, 25, 31, 351, 351}, {1, 1, 1, 1, 1, 0}};

  auto struct_1 = cudf::test::structs_column_wrapper{{names_col, ages_col}, {1, 1, 1, 1, 0, 1}};

  auto is_human_col = cudf::test::fixed_width_column_wrapper<bool>{
    {true, true, false, false, false, false}, {1, 1, 0, 1, 1, 0}};

  auto struct_2 =
    cudf::test::structs_column_wrapper{{is_human_col, struct_1}, {0, 1, 1, 1, 1, 1}}.release();

  auto expected = table_view({*struct_2});

  auto filepath = temp_env->get_temp_filepath("Struct.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath));
  cudf::io::read_parquet(read_args);
}

// custom data sink that supports device writes. uses plain file io.
class custom_test_data_sink : public cudf::io::data_sink {
 public:
  explicit custom_test_data_sink(std::string const& filepath)
  {
    outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
    CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
  }

  virtual ~custom_test_data_sink() { flush(); }

  void host_write(void const* data, size_t size) override
  {
    outfile_.write(static_cast<char const*>(data), size);
  }

  [[nodiscard]] bool supports_device_write() const override { return true; }

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
      outfile_.write(ptr, size);
      CUDF_CUDA_TRY(cudaFreeHost(ptr));
    });
  }

  void flush() override { outfile_.flush(); }

  size_t bytes_written() override { return outfile_.tellp(); }

 private:
  std::ofstream outfile_;
};

TEST_F(ParquetWriterTest, CustomDataSink)
{
  auto filepath = temp_env->get_temp_filepath("CustomDataSink.parquet");
  custom_test_data_sink custom_sink(filepath);

  srand(31337);
  auto expected = create_random_fixed_table<int>(5, 10, false);

  // write out using the custom sink
  {
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected);
    cudf::io::write_parquet(args);
  }

  // write out using a memmapped sink
  std::vector<char> buf_sink;
  {
    cudf::io::parquet_writer_options args =
      cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&buf_sink}, *expected);
    cudf::io::write_parquet(args);
  }

  // read them back in and make sure everything matches

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());

  cudf::io::parquet_reader_options buf_args = cudf::io::parquet_reader_options::builder(
    cudf::io::source_info{buf_sink.data(), buf_sink.size()});
  auto buf_tbl = cudf::io::read_parquet(buf_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(buf_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterTest, DeviceWriteLargeishFile)
{
  auto filepath = temp_env->get_temp_filepath("DeviceWriteLargeishFile.parquet");
  custom_test_data_sink custom_sink(filepath);

  // exercises multiple rowgroups
  srand(31337);
  auto expected = create_random_fixed_table<int>(4, 1024 * 1024, false);

  // write out using the custom sink (which uses device writes)
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&custom_sink}, *expected)
      .row_group_size_rows(128 * 1024);
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options custom_args =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto custom_tbl = cudf::io::read_parquet(custom_args);
  CUDF_TEST_EXPECT_TABLES_EQUAL(custom_tbl.tbl->view(), expected->view());
}

TEST_F(ParquetWriterTest, PartitionedWrite)
{
  auto source = create_compressible_fixed_table<int>(16, 4 * 1024 * 1024, 1000, false);

  auto filepath1 = temp_env->get_temp_filepath("PartitionedWrite1.parquet");
  auto filepath2 = temp_env->get_temp_filepath("PartitionedWrite2.parquet");

  auto partition1 = cudf::io::partition_info{10, 1024 * 1024};
  auto partition2 = cudf::io::partition_info{20 * 1024 + 7, 3 * 1024 * 1024};

  auto expected1 =
    cudf::slice(*source, {partition1.start_row, partition1.start_row + partition1.num_rows});
  auto expected2 =
    cudf::slice(*source, {partition2.start_row, partition2.start_row + partition2.num_rows});

  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(
      cudf::io::sink_info(std::vector<std::string>{filepath1, filepath2}), *source)
      .partitions({partition1, partition2})
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(args);

  auto result1 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath1)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected1, result1.tbl->view());

  auto result2 = cudf::io::read_parquet(
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(filepath2)));
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected2, result2.tbl->view());
}

template <typename T>
std::string create_parquet_file(int num_cols)
{
  srand(31337);
  auto const table = create_random_fixed_table<T>(num_cols, 10, true);
  auto const filepath =
    temp_env->get_temp_filepath(typeid(T).name() + std::to_string(num_cols) + ".parquet");
  cudf::io::parquet_writer_options const out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table->view());
  cudf::io::write_parquet(out_opts);
  return filepath;
}

TEST_F(ParquetWriterTest, MultipleMismatchedSources)
{
  auto const int5file = create_parquet_file<int>(5);
  {
    auto const float5file = create_parquet_file<float>(5);
    std::vector<std::string> files{int5file, float5file};
    cudf::io::parquet_reader_options const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{files});
    EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  }
  {
    auto const int10file = create_parquet_file<int>(10);
    std::vector<std::string> files{int5file, int10file};
    cudf::io::parquet_reader_options const read_opts =
      cudf::io::parquet_reader_options::builder(cudf::io::source_info{files});
    EXPECT_THROW(cudf::io::read_parquet(read_opts), cudf::logic_error);
  }
}

TEST_F(ParquetWriterTest, Slice)
{
  auto col =
    cudf::test::fixed_width_column_wrapper<int>{{1, 2, 3, 4, 5}, {true, true, true, false, true}};
  std::vector<cudf::size_type> indices{2, 5};
  std::vector<cudf::column_view> result = cudf::slice(col, indices);
  cudf::table_view tbl{result};

  auto filepath = temp_env->get_temp_filepath("Slice.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, tbl);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto read_table = cudf::io::read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUIVALENT(read_table.tbl->view(), tbl);
}

TEST_F(ParquetWriterTest, DecimalWrite)
{
  constexpr cudf::size_type num_rows = 500;
  auto seq_col0                      = random_values<int32_t>(num_rows);
  auto seq_col1                      = random_values<int64_t>(num_rows);

  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  auto col0 = cudf::test::fixed_point_column_wrapper<int32_t>{
    seq_col0.begin(), seq_col0.end(), valids, numeric::scale_type{5}};
  auto col1 = cudf::test::fixed_point_column_wrapper<int64_t>{
    seq_col1.begin(), seq_col1.end(), valids, numeric::scale_type{-9}};

  auto table = table_view({col0, col1});

  auto filepath = temp_env->get_temp_filepath("DecimalWrite.parquet");
  cudf::io::parquet_writer_options args =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table);

  cudf::io::table_input_metadata expected_metadata(table);

  // verify failure if too small a precision is given
  expected_metadata.column_metadata[0].set_decimal_precision(7);
  expected_metadata.column_metadata[1].set_decimal_precision(1);
  args.set_metadata(expected_metadata);
  EXPECT_THROW(cudf::io::write_parquet(args), cudf::logic_error);

  // verify success if equal precision is given
  expected_metadata.column_metadata[0].set_decimal_precision(7);
  expected_metadata.column_metadata[1].set_decimal_precision(9);
  args.set_metadata(std::move(expected_metadata));
  cudf::io::write_parquet(args);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(*result.tbl, table);
}

TEST_F(ParquetWriterTest, RowGroupSizeInvalid)
{
  auto const unused_table = std::make_unique<table>();
  std::vector<char> out_buffer;

  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .row_group_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .row_group_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                         unused_table->view())
                 .max_page_size_bytes(0xFFFF'FFFFUL),
               cudf::logic_error);

  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .row_group_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_rows(0),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .row_group_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_bytes(3 << 8),
               cudf::logic_error);
  EXPECT_THROW(cudf::io::chunked_parquet_writer_options::builder(cudf::io::sink_info(&out_buffer))
                 .max_page_size_bytes(0xFFFF'FFFFUL),
               cudf::logic_error);
}

TEST_F(ParquetWriterTest, RowGroupPageSizeMatch)
{
  auto const unused_table = std::make_unique<table>();
  std::vector<char> out_buffer;

  auto options = cudf::io::parquet_writer_options::builder(cudf::io::sink_info(&out_buffer),
                                                           unused_table->view())
                   .row_group_size_bytes(128 * 1024)
                   .max_page_size_bytes(512 * 1024)
                   .row_group_size_rows(10000)
                   .max_page_size_rows(20000)
                   .build();
  EXPECT_EQ(options.get_row_group_size_bytes(), options.get_max_page_size_bytes());
  EXPECT_EQ(options.get_row_group_size_rows(), options.get_max_page_size_rows());
}

TEST_F(ParquetWriterTest, EmptyList)
{
  auto L1 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("EmptyList.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));

  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  using lcw     = cudf::test::lists_column_wrapper<int64_t>;
  auto expected = lcw{lcw{}, lcw{}, lcw{}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), expected);
}

TEST_F(ParquetWriterTest, DeepEmptyList)
{
  // Make a list column LLLi st only L is valid and LLi are all null. This tests whether we can
  // handle multiple nullptr offsets

  auto L2 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});
  auto L1 = cudf::make_lists_column(
    0, cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)), std::move(L2), 0, {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("DeepEmptyList.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));

  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), *L0);
}

TEST_F(ParquetWriterTest, EmptyListWithStruct)
{
  auto L2 = cudf::make_lists_column(0,
                                    cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)),
                                    cudf::make_empty_column(cudf::data_type{cudf::type_id::INT64}),
                                    0,
                                    {});

  auto children = std::vector<std::unique_ptr<cudf::column>>{};
  children.push_back(std::move(L2));
  auto S2 = cudf::make_structs_column(0, std::move(children), 0, {});
  auto L1 = cudf::make_lists_column(
    0, cudf::make_empty_column(cudf::data_type(cudf::type_id::INT32)), std::move(S2), 0, {});
  auto L0 = cudf::make_lists_column(
    3, cudf::test::fixed_width_column_wrapper<int32_t>{0, 0, 0, 0}.release(), std::move(L1), 0, {});

  auto filepath = temp_env->get_temp_filepath("EmptyListWithStruct.parquet");
  cudf::io::write_parquet(cudf::io::parquet_writer_options_builder(cudf::io::sink_info(filepath),
                                                                   cudf::table_view({*L0})));
  auto result = cudf::io::read_parquet(
    cudf::io::parquet_reader_options_builder(cudf::io::source_info(filepath)));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tbl->view().column(0), *L0);
}

TEST_F(ParquetWriterTest, CheckPageRows)
{
  auto sequence = thrust::make_counting_iterator(0);

  constexpr auto page_rows = 5000;
  constexpr auto num_rows  = 2 * page_rows;
  column_wrapper<int> col(sequence, sequence + num_rows, no_nulls());

  auto expected = table_view{{col}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRows.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(page_rows);
  cudf::io::write_parquet(out_opts);

  // check first page header and make sure it has only page_rows values
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);
  ASSERT_EQ(fmd.row_groups[0].columns.size(), 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_GT(first_chunk.data_page_offset, 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::detail::PageHeader), 0});

  EXPECT_EQ(ph.data_page_header.num_values, page_rows);
}

TEST_F(ParquetWriterTest, CheckPageRowsAdjusted)
{
  // enough for a few pages with the default 20'000 rows/page
  constexpr auto rows_per_page = 20'000;
  constexpr auto num_rows      = 3 * rows_per_page;
  const std::string s1(32, 'a');
  auto col0_elements =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s1; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + num_rows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRowsAdjusted.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_size_rows(rows_per_page);
  cudf::io::write_parquet(out_opts);

  // check first page header and make sure it has only page_rows values
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_GT(fmd.row_groups.size(), 0);
  ASSERT_EQ(fmd.row_groups[0].columns.size(), 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_GT(first_chunk.data_page_offset, 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::detail::PageHeader), 0});

  EXPECT_LE(ph.data_page_header.num_values, rows_per_page);
}

TEST_F(ParquetWriterTest, CheckPageRowsTooSmall)
{
  constexpr auto rows_per_page = 1'000;
  constexpr auto fragment_size = 5'000;
  constexpr auto num_rows      = 3 * rows_per_page;
  const std::string s1(32, 'a');
  auto col0_elements =
    cudf::detail::make_counting_transform_iterator(0, [&](auto i) { return s1; });
  auto col0 = cudf::test::strings_column_wrapper(col0_elements, col0_elements + num_rows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("CheckPageRowsTooSmall.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .max_page_fragment_size(fragment_size)
      .max_page_size_rows(rows_per_page);
  cudf::io::write_parquet(out_opts);

  // check that file is written correctly when rows/page < fragment size
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  ASSERT_TRUE(fmd.row_groups.size() > 0);
  ASSERT_TRUE(fmd.row_groups[0].columns.size() == 1);
  auto const& first_chunk = fmd.row_groups[0].columns[0].meta_data;
  ASSERT_TRUE(first_chunk.data_page_offset > 0);

  // read first data page header.  sizeof(PageHeader) is not exact, but the thrift encoded
  // version should be smaller than size of the struct.
  auto const ph = read_page_header(
    source, {first_chunk.data_page_offset, sizeof(cudf::io::parquet::detail::PageHeader), 0});

  // there should be only one page since the fragment size is larger than rows_per_page
  EXPECT_EQ(ph.data_page_header.num_values, num_rows);
}

TEST_F(ParquetWriterTest, Decimal32Stats)
{
  // check that decimal64 min and max statistics are written properly
  std::vector<uint8_t> expected_min{0, 0, 0xb2, 0xa1};
  std::vector<uint8_t> expected_max{0xb2, 0xa1, 0, 0};

  int32_t val0 = 0xa1b2;
  int32_t val1 = val0 << 16;
  column_wrapper<numeric::decimal32> col0{{numeric::decimal32(val0, numeric::scale_type{0}),
                                           numeric::decimal32(val1, numeric::scale_type{0})}};

  auto expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("Decimal32Stats.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  auto const stats = get_statistics(fmd.row_groups[0].columns[0]);

  EXPECT_EQ(expected_min, stats.min_value);
  EXPECT_EQ(expected_max, stats.max_value);
}

TEST_F(ParquetWriterTest, Decimal64Stats)
{
  // check that decimal64 min and max statistics are written properly
  std::vector<uint8_t> expected_min{0, 0, 0, 0, 0xd4, 0xc3, 0xb2, 0xa1};
  std::vector<uint8_t> expected_max{0xd4, 0xc3, 0xb2, 0xa1, 0, 0, 0, 0};

  int64_t val0 = 0xa1b2'c3d4UL;
  int64_t val1 = val0 << 32;
  column_wrapper<numeric::decimal64> col0{{numeric::decimal64(val0, numeric::scale_type{0}),
                                           numeric::decimal64(val1, numeric::scale_type{0})}};

  auto expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("Decimal64Stats.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  auto const stats = get_statistics(fmd.row_groups[0].columns[0]);

  EXPECT_EQ(expected_min, stats.min_value);
  EXPECT_EQ(expected_max, stats.max_value);
}

TEST_F(ParquetWriterTest, Decimal128Stats)
{
  // check that decimal128 min and max statistics are written in network byte order
  // this is negative, so should be the min
  std::vector<uint8_t> expected_min{
    0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> expected_max{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6};

  __int128_t val0 = 0xa1b2'c3d4'e5f6ULL;
  __int128_t val1 = val0 << 80;
  column_wrapper<numeric::decimal128> col0{{numeric::decimal128(val0, numeric::scale_type{0}),
                                            numeric::decimal128(val1, numeric::scale_type{0})}};

  auto expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("Decimal128Stats.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  auto const stats = get_statistics(fmd.row_groups[0].columns[0]);

  EXPECT_EQ(expected_min, stats.min_value);
  EXPECT_EQ(expected_max, stats.max_value);
}

TEST_F(ParquetWriterTest, CheckColumnIndexTruncation)
{
  char const* coldata[] = {
    // in-range 7 bit.  should truncate to "yyyyyyyz"
    "yyyyyyyyy",
    // max 7 bit. should truncate to "x7fx7fx7fx7fx7fx7fx7fx80", since it's
    // considered binary, not UTF-8.  If UTF-8 it should not truncate.
    "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
    // max binary.  this should not truncate
    "\xff\xff\xff\xff\xff\xff\xff\xff\xff",
    // in-range 2-byte UTF8 (U+00E9). should truncate to "éééê"
    "ééééé",
    // max 2-byte UTF8 (U+07FF). should not truncate
    "߿߿߿߿߿",
    // in-range 3-byte UTF8 (U+0800). should truncate to "ࠀࠁ"
    "ࠀࠀࠀ",
    // max 3-byte UTF8 (U+FFFF). should not truncate
    "\xef\xbf\xbf\xef\xbf\xbf\xef\xbf\xbf",
    // in-range 4-byte UTF8 (U+10000). should truncate to "𐀀𐀁"
    "𐀀𐀀𐀀",
    // max unicode (U+10FFFF). should truncate to \xf4\x8f\xbf\xbf\xf4\x90\x80\x80,
    // which is no longer valid unicode, but is still ok UTF-8???
    "\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf",
    // max 4-byte UTF8 (U+1FFFFF). should not truncate
    "\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf"};

  // NOTE: UTF8 min is initialized with 0xf7bfbfbf. Binary values larger
  // than that will not become minimum value (when written as UTF-8).
  char const* truncated_min[] = {"yyyyyyyy",
                                 "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x7f",
                                 "\xf7\xbf\xbf\xbf",
                                 "éééé",
                                 "߿߿߿߿",
                                 "ࠀࠀ",
                                 "\xef\xbf\xbf\xef\xbf\xbf",
                                 "𐀀𐀀",
                                 "\xf4\x8f\xbf\xbf\xf4\x8f\xbf\xbf",
                                 "\xf7\xbf\xbf\xbf"};

  char const* truncated_max[] = {"yyyyyyyz",
                                 "\x7f\x7f\x7f\x7f\x7f\x7f\x7f\x80",
                                 "\xff\xff\xff\xff\xff\xff\xff\xff\xff",
                                 "éééê",
                                 "߿߿߿߿߿",
                                 "ࠀࠁ",
                                 "\xef\xbf\xbf\xef\xbf\xbf\xef\xbf\xbf",
                                 "𐀀𐀁",
                                 "\xf4\x8f\xbf\xbf\xf4\x90\x80\x80",
                                 "\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf\xf7\xbf\xbf\xbf"};

  auto cols = [&]() {
    using string_wrapper = column_wrapper<cudf::string_view>;
    std::vector<std::unique_ptr<column>> cols;
    for (auto const str : coldata) {
      cols.push_back(string_wrapper{str}.release());
    }
    return cols;
  }();
  auto expected = std::make_unique<table>(std::move(cols));

  auto const filepath = temp_env->get_temp_filepath("CheckColumnIndexTruncation.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected->view())
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .column_index_truncate_length(8);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value.value(), ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value.value(), ptype, ctype) >= 0);

      // check that truncated values == expected
      EXPECT_EQ(memcmp(ci.min_values[0].data(), truncated_min[c], ci.min_values[0].size()), 0);
      EXPECT_EQ(memcmp(ci.max_values[0].data(), truncated_max[c], ci.max_values[0].size()), 0);
    }
  }
}

TEST_F(ParquetWriterTest, BinaryColumnIndexTruncation)
{
  std::vector<uint8_t> truncated_min[] = {{0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe},
                                          {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
                                          {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  std::vector<uint8_t> truncated_max[] = {{0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xff},
                                          {0xff},
                                          {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  cudf::test::lists_column_wrapper<uint8_t> col0{
    {0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe, 0xfe}};
  cudf::test::lists_column_wrapper<uint8_t> col1{
    {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};
  cudf::test::lists_column_wrapper<uint8_t> col2{
    {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}};

  auto expected = table_view{{col0, col1, col2}};

  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_binary0").set_output_as_binary(true);
  output_metadata.column_metadata[1].set_name("col_binary1").set_output_as_binary(true);
  output_metadata.column_metadata[2].set_name("col_binary2").set_output_as_binary(true);

  auto const filepath = temp_env->get_temp_filepath("BinaryColumnIndexTruncation.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata))
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .column_index_truncate_length(8);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  for (size_t r = 0; r < fmd.row_groups.size(); r++) {
    auto const& rg = fmd.row_groups[r];
    for (size_t c = 0; c < rg.columns.size(); c++) {
      auto const& chunk = rg.columns[c];

      auto const ci    = read_column_index(source, chunk);
      auto const stats = get_statistics(chunk);

      // check trunc(page.min) <= stats.min && trun(page.max) >= stats.max
      auto const ptype = fmd.schema[c + 1].type;
      auto const ctype = fmd.schema[c + 1].converted_type;
      ASSERT_TRUE(stats.min_value.has_value());
      ASSERT_TRUE(stats.max_value.has_value());
      EXPECT_TRUE(compare_binary(ci.min_values[0], stats.min_value.value(), ptype, ctype) <= 0);
      EXPECT_TRUE(compare_binary(ci.max_values[0], stats.max_value.value(), ptype, ctype) >= 0);

      // check that truncated values == expected
      EXPECT_EQ(ci.min_values[0], truncated_min[c]);
      EXPECT_EQ(ci.max_values[0], truncated_max[c]);
    }
  }
}

TEST_F(ParquetWriterTest, ByteArrayStats)
{
  // check that byte array min and max statistics are written as expected. If a byte array is
  // written as a string, max utf8 is 0xf7bfbfbf and so the minimum value will be set to that value
  // instead of a potential minimum higher than that.
  std::vector<uint8_t> expected_col0_min{0xf0};
  std::vector<uint8_t> expected_col0_max{0xf0, 0xf5, 0xf5};
  std::vector<uint8_t> expected_col1_min{0xfe, 0xfe, 0xfe};
  std::vector<uint8_t> expected_col1_max{0xfe, 0xfe, 0xfe};

  cudf::test::lists_column_wrapper<uint8_t> list_int_col0{
    {0xf0}, {0xf0, 0xf5, 0xf3}, {0xf0, 0xf5, 0xf5}};
  cudf::test::lists_column_wrapper<uint8_t> list_int_col1{
    {0xfe, 0xfe, 0xfe}, {0xfe, 0xfe, 0xfe}, {0xfe, 0xfe, 0xfe}};

  auto expected = table_view{{list_int_col0, list_int_col1}};
  cudf::io::table_input_metadata output_metadata(expected);
  output_metadata.column_metadata[0].set_name("col_binary0").set_output_as_binary(true);
  output_metadata.column_metadata[1].set_name("col_binary1").set_output_as_binary(true);

  auto filepath = temp_env->get_temp_filepath("ByteArrayStats.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(output_metadata));
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath})
      .set_column_schema({{}, {}});
  auto result = cudf::io::read_parquet(in_opts);

  auto source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);

  EXPECT_EQ(fmd.schema[1].type, cudf::io::parquet::detail::Type::BYTE_ARRAY);
  EXPECT_EQ(fmd.schema[2].type, cudf::io::parquet::detail::Type::BYTE_ARRAY);

  auto const stats0 = get_statistics(fmd.row_groups[0].columns[0]);
  auto const stats1 = get_statistics(fmd.row_groups[0].columns[1]);

  EXPECT_EQ(expected_col0_min, stats0.min_value);
  EXPECT_EQ(expected_col0_max, stats0.max_value);
  EXPECT_EQ(expected_col1_min, stats1.min_value);
  EXPECT_EQ(expected_col1_max, stats1.max_value);
}

TEST_F(ParquetWriterTest, SingleValueDictionaryTest)
{
  constexpr unsigned int expected_bits = 1;
  constexpr unsigned int nrows         = 1'000'000U;

  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("SingleValueDictionaryTest.parquet");
  // set row group size so that there will be only one row group
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .row_group_size_rows(nrows);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict());

  // and check that the correct number of bits was used
  auto const oi    = read_offset_index(source, fmd.row_groups[0].columns[0]);
  auto const nbits = read_dict_bits(source, oi.page_locations[0]);
  EXPECT_EQ(nbits, expected_bits);
}

TEST_F(ParquetWriterTest, DictionaryNeverTest)
{
  constexpr unsigned int nrows = 1'000U;

  // only one value, so would normally use dictionary
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0     = cudf::test::strings_column_wrapper(elements, elements + nrows);
  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryNeverTest.parquet");
  // no compression so we can easily read page data
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was not used
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd]() {
    for (auto enc : fmd.row_groups[0].columns[0].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_FALSE(used_dict());
}

TEST_F(ParquetWriterTest, DictionaryAdaptiveTest)
{
  constexpr unsigned int nrows = 65'536U;
  // cardinality is chosen to result in a dictionary > 1MB in size
  constexpr unsigned int cardinality = 32'768U;

  // single value will have a small dictionary
  auto elements0 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  // high cardinality will have a large dictionary
  auto elements1  = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col1 = cudf::test::strings_column_wrapper(elements1, elements1 + nrows);

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryAdaptiveTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ADAPTIVE);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used as expected. col0 should use one,
  // col1 should not.
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd](int col) {
    for (auto enc : fmd.row_groups[0].columns[col].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict(0));
  EXPECT_FALSE(used_dict(1));
}

TEST_F(ParquetWriterTest, DictionaryAlwaysTest)
{
  constexpr unsigned int nrows = 65'536U;
  // cardinality is chosen to result in a dictionary > 1MB in size
  constexpr unsigned int cardinality = 32'768U;

  // single value will have a small dictionary
  auto elements0 = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return "a unique string value suffixed with 1"; });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  // high cardinality will have a large dictionary
  auto elements1  = cudf::detail::make_counting_transform_iterator(0, [cardinality](auto i) {
    return "a unique string value suffixed with " + std::to_string(i % cardinality);
  });
  auto const col1 = cudf::test::strings_column_wrapper(elements1, elements1 + nrows);

  auto const expected = table_view{{col0, col1}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryAlwaysTest.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());

  // make sure dictionary was used for both columns
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;

  read_footer(source, &fmd);
  auto used_dict = [&fmd](int col) {
    for (auto enc : fmd.row_groups[0].columns[col].meta_data.encodings) {
      if (enc == cudf::io::parquet::detail::Encoding::PLAIN_DICTIONARY or
          enc == cudf::io::parquet::detail::Encoding::RLE_DICTIONARY) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(used_dict(0));
  EXPECT_TRUE(used_dict(1));
}

TEST_F(ParquetWriterTest, DictionaryPageSizeEst)
{
  // one page
  constexpr unsigned int nrows = 20'000U;

  // this test is creating a pattern of repeating then non-repeating values to trigger
  // a "worst-case" for page size estimation in the presence of a dictionary. have confirmed
  // that this fails for values over 16 in the final term of `max_RLE_page_size()`.
  // The output of the iterator will be 'CCCCCRRRRRCCCCCRRRRR...` where 'C' is a changing
  // value, and 'R' repeats. The encoder will turn this into a literal run of 8 values
  // (`CCCCCRRR`) followed by a repeated run of 2 (`RR`). This pattern then repeats, getting
  // as close as possible to a condition of repeated 8 value literal runs.
  auto elements0  = cudf::detail::make_counting_transform_iterator(0, [](auto i) {
    if ((i / 5) % 2 == 1) {
      return std::string("non-unique string");
    } else {
      return "a unique string value suffixed with " + std::to_string(i);
    }
  });
  auto const col0 = cudf::test::strings_column_wrapper(elements0, elements0 + nrows);

  auto const expected = table_view{{col0}};

  auto const filepath = temp_env->get_temp_filepath("DictionaryPageSizeEst.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::ZSTD)
      .dictionary_policy(cudf::io::dictionary_policy::ALWAYS);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options default_in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result = cudf::io::read_parquet(default_in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetWriterTest, UserNullability)
{
  auto weight_col = cudf::test::fixed_width_column_wrapper<float>{{57.5, 51.1, 15.3}};
  auto ages_col   = cudf::test::fixed_width_column_wrapper<int32_t>{{30, 27, 5}};
  auto struct_col = cudf::test::structs_column_wrapper{weight_col, ages_col};

  auto expected = table_view({struct_col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);
  expected_metadata.column_metadata[0].child(0).set_nullability(true);

  auto filepath = temp_env->get_temp_filepath("SingleWriteNullable.parquet");
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(read_opts);

  EXPECT_FALSE(result.tbl->view().column(0).nullable());
  EXPECT_TRUE(result.tbl->view().column(0).child(0).nullable());
  EXPECT_FALSE(result.tbl->view().column(0).child(1).nullable());
}

TEST_F(ParquetWriterTest, UserNullabilityInvalid)
{
  auto valids =
    cudf::detail::make_counting_transform_iterator(0, [&](int index) { return index % 2; });
  auto col      = cudf::test::fixed_width_column_wrapper<double>{{57.5, 51.1, 15.3}, valids};
  auto expected = table_view({col});

  auto filepath = temp_env->get_temp_filepath("SingleWriteNullableInvalid.parquet");
  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected);
  // Should work without the nullability option
  EXPECT_NO_THROW(cudf::io::write_parquet(write_opts));

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);
  write_opts.set_metadata(std::move(expected_metadata));
  // Can't write a column with nulls as not nullable
  EXPECT_THROW(cudf::io::write_parquet(write_opts), cudf::logic_error);
}

TEST_F(ParquetWriterTest, CompStats)
{
  auto table = create_random_fixed_table<int>(1, 100000, true);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer}, table->view())
      .compression_statistics(stats);
  cudf::io::write_parquet(opts);

  EXPECT_NE(stats->num_compressed_bytes(), 0);
  EXPECT_EQ(stats->num_failed_bytes(), 0);
  EXPECT_EQ(stats->num_skipped_bytes(), 0);
  EXPECT_FALSE(std::isnan(stats->compression_ratio()));
}

TEST_F(ParquetWriterTest, CompStatsEmptyTable)
{
  auto table_no_rows = create_random_fixed_table<int>(20, 0, false);

  auto const stats = std::make_shared<cudf::io::writer_compression_statistics>();

  std::vector<char> unused_buffer;
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{&unused_buffer},
                                              table_no_rows->view())
      .compression_statistics(stats);
  cudf::io::write_parquet(opts);

  expect_compression_stats_empty(stats);
}

TEST_F(ParquetWriterTest, NoNullsAsNonNullable)
{
  column_wrapper<int32_t> col{{1, 2, 3}, no_nulls()};
  table_view expected({col});

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_nullability(false);

  auto filepath = temp_env->get_temp_filepath("NonNullable.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(std::move(expected_metadata));
  // Writer should be able to write a column without nulls as non-nullable
  EXPECT_NO_THROW(cudf::io::write_parquet(out_opts));
}

TEST_F(ParquetWriterTest, TimestampMicrosINT96NoOverflow)
{
  using namespace cuda::std::chrono;
  using namespace cudf::io;

  column_wrapper<cudf::timestamp_us> big_ts_col{
    sys_days{year{3023} / month{7} / day{14}} + 7h + 38min + 45s + 418688us,
    sys_days{year{723} / month{3} / day{21}} + 14h + 20min + 13s + microseconds{781ms}};

  table_view expected({big_ts_col});
  auto filepath = temp_env->get_temp_filepath("BigINT96Timestamp.parquet");

  auto const out_opts =
    parquet_writer_options::builder(sink_info{filepath}, expected).int96_timestamps(true).build();
  write_parquet(out_opts);

  auto const in_opts = parquet_reader_options::builder(source_info(filepath))
                         .timestamp_type(cudf::data_type(cudf::type_id::TIMESTAMP_MICROSECONDS))
                         .build();
  auto const result = read_parquet(in_opts);

  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

TEST_F(ParquetWriterTest, PreserveNullability)
{
  constexpr auto num_rows = 100;

  auto const col0_data = random_values<int32_t>(num_rows);
  auto const col1_data = random_values<int32_t>(num_rows);

  auto const col0_validity = cudf::test::iterators::no_nulls();
  auto const col1_validity =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });

  column_wrapper<int32_t> col0{col0_data.begin(), col0_data.end(), col0_validity};
  column_wrapper<int32_t> col1{col1_data.begin(), col1_data.end(), col1_validity};
  auto const col2 = make_parquet_list_list_col<int>(0, num_rows, 5, 8, true);

  auto const expected = table_view{{col0, col1, *col2}};

  cudf::io::table_input_metadata expected_metadata(expected);
  expected_metadata.column_metadata[0].set_name("mandatory");
  expected_metadata.column_metadata[0].set_nullability(false);
  expected_metadata.column_metadata[1].set_name("optional");
  expected_metadata.column_metadata[1].set_nullability(true);
  expected_metadata.column_metadata[2].set_name("lists");
  expected_metadata.column_metadata[2].set_nullability(true);
  // offsets is a cudf thing that's not part of the parquet schema so it won't have nullability set
  expected_metadata.column_metadata[2].child(0).set_name("offsets");
  expected_metadata.column_metadata[2].child(1).set_name("element");
  expected_metadata.column_metadata[2].child(1).set_nullability(false);
  expected_metadata.column_metadata[2].child(1).child(0).set_name("offsets");
  expected_metadata.column_metadata[2].child(1).child(1).set_name("element");
  expected_metadata.column_metadata[2].child(1).child(1).set_nullability(true);

  auto const filepath = temp_env->get_temp_filepath("PreserveNullability.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .metadata(expected_metadata);

  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options const in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto const result        = cudf::io::read_parquet(in_opts);
  auto const read_metadata = cudf::io::table_input_metadata{result.metadata};

  // test that expected_metadata matches read_metadata
  std::function<void(cudf::io::column_in_metadata, cudf::io::column_in_metadata)>
    compare_names_and_nullability = [&](auto lhs, auto rhs) {
      EXPECT_EQ(lhs.get_name(), rhs.get_name());
      ASSERT_EQ(lhs.is_nullability_defined(), rhs.is_nullability_defined());
      if (lhs.is_nullability_defined()) { EXPECT_EQ(lhs.nullable(), rhs.nullable()); }
      ASSERT_EQ(lhs.num_children(), rhs.num_children());
      for (int i = 0; i < lhs.num_children(); ++i) {
        compare_names_and_nullability(lhs.child(i), rhs.child(i));
      }
    };

  ASSERT_EQ(expected_metadata.column_metadata.size(), read_metadata.column_metadata.size());

  for (size_t i = 0; i < expected_metadata.column_metadata.size(); ++i) {
    compare_names_and_nullability(expected_metadata.column_metadata[i],
                                  read_metadata.column_metadata[i]);
  }
}

TEST_F(ParquetWriterTest, EmptyMinStringStatistics)
{
  char const* const min_val = "";
  char const* const max_val = "zzz";
  std::vector<char const*> strings{min_val, max_val, "pining", "for", "the", "fjords"};

  column_wrapper<cudf::string_view> string_col{strings.begin(), strings.end()};
  auto const output   = table_view{{string_col}};
  auto const filepath = temp_env->get_temp_filepath("EmptyMinStringStatistics.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, output);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;
  read_footer(source, &fmd);

  ASSERT_TRUE(fmd.row_groups.size() > 0);
  ASSERT_TRUE(fmd.row_groups[0].columns.size() > 0);
  auto const& chunk = fmd.row_groups[0].columns[0];
  auto const stats  = get_statistics(chunk);

  ASSERT_TRUE(stats.min_value.has_value());
  ASSERT_TRUE(stats.max_value.has_value());
  auto const min_value = std::string{reinterpret_cast<char const*>(stats.min_value.value().data()),
                                     stats.min_value.value().size()};
  auto const max_value = std::string{reinterpret_cast<char const*>(stats.max_value.value().data()),
                                     stats.max_value.value().size()};
  EXPECT_EQ(min_value, std::string(min_val));
  EXPECT_EQ(max_value, std::string(max_val));
}

TEST_F(ParquetWriterTest, RowGroupMetadata)
{
  using column_type      = int;
  constexpr int num_rows = 1'000;
  auto const ones        = thrust::make_constant_iterator(1);
  auto const col =
    cudf::test::fixed_width_column_wrapper<column_type>{ones, ones + num_rows, no_nulls()};
  auto const table = table_view({col});

  auto const filepath = temp_env->get_temp_filepath("RowGroupMetadata.parquet");
  // force PLAIN encoding to make size calculation easier
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_parquet(opts);

  // check row group metadata to make sure total_byte_size is the uncompressed value
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;
  read_footer(source, &fmd);

  ASSERT_GT(fmd.row_groups.size(), 0);
  EXPECT_GE(fmd.row_groups[0].total_byte_size,
            static_cast<int64_t>(num_rows * sizeof(column_type)));
}

TEST_F(ParquetWriterTest, UserRequestedDictFallback)
{
  constexpr int num_rows = 100;
  constexpr char const* big_string =
    "a "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "long string";

  auto const max_dict_size = strlen(big_string) * num_rows / 2;

  auto elements1 = cudf::detail::make_counting_transform_iterator(
    0, [big_string](auto i) { return big_string + std::to_string(i); });
  auto const col1  = cudf::test::strings_column_wrapper(elements1, elements1 + num_rows);
  auto const table = table_view({col1});

  cudf::io::table_input_metadata table_metadata(table);
  table_metadata.column_metadata[0]
    .set_name("big_strings")
    .set_encoding(cudf::io::column_encoding::DICTIONARY)
    .set_nullability(false);

  auto const filepath = temp_env->get_temp_filepath("UserRequestedDictFallback.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table)
      .metadata(table_metadata)
      .max_dictionary_size(max_dict_size);
  cudf::io::write_parquet(opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;
  read_footer(source, &fmd);

  // encoding should have fallen back to PLAIN
  EXPECT_EQ(fmd.row_groups[0].columns[0].meta_data.encodings[0],
            cudf::io::parquet::detail::Encoding::PLAIN);
}

TEST_F(ParquetWriterTest, UserRequestedEncodings)
{
  using cudf::io::column_encoding;
  using cudf::io::parquet::detail::Encoding;
  constexpr int num_rows = 500;

  auto const ones = thrust::make_constant_iterator(1);
  auto const col =
    cudf::test::fixed_width_column_wrapper<int32_t>{ones, ones + num_rows, no_nulls()};

  auto const strings = thrust::make_constant_iterator("string");
  auto const string_col =
    cudf::test::strings_column_wrapper(strings, strings + num_rows, no_nulls());

  auto const table = table_view({col,
                                 col,
                                 col,
                                 col,
                                 col,
                                 col,
                                 string_col,
                                 string_col,
                                 string_col,
                                 string_col,
                                 string_col,
                                 string_col});

  cudf::io::table_input_metadata table_metadata(table);

  auto const set_meta = [&table_metadata](int idx, std::string const& name, column_encoding enc) {
    table_metadata.column_metadata[idx].set_name(name).set_encoding(enc);
  };

  set_meta(0, "int_plain", column_encoding::PLAIN);
  set_meta(1, "int_dict", column_encoding::DICTIONARY);
  set_meta(2, "int_db", column_encoding::DELTA_BINARY_PACKED);
  set_meta(3, "int_dlba", column_encoding::DELTA_LENGTH_BYTE_ARRAY);
  set_meta(4, "int_dba", column_encoding::DELTA_BYTE_ARRAY);
  table_metadata.column_metadata[5].set_name("int_none");

  set_meta(6, "string_plain", column_encoding::PLAIN);
  set_meta(7, "string_dict", column_encoding::DICTIONARY);
  set_meta(8, "string_dlba", column_encoding::DELTA_LENGTH_BYTE_ARRAY);
  set_meta(9, "string_dba", column_encoding::DELTA_BYTE_ARRAY);
  set_meta(10, "string_db", column_encoding::DELTA_BINARY_PACKED);
  table_metadata.column_metadata[11].set_name("string_none");

  for (auto& col_meta : table_metadata.column_metadata) {
    col_meta.set_nullability(false);
  }

  auto const filepath = temp_env->get_temp_filepath("UserRequestedEncodings.parquet");
  cudf::io::parquet_writer_options opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, table)
      .metadata(table_metadata)
      .stats_level(cudf::io::statistics_freq::STATISTICS_COLUMN)
      .compression(cudf::io::compression_type::ZSTD);
  cudf::io::write_parquet(opts);

  // check page headers to make sure each column is encoded with the appropriate encoder
  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;
  read_footer(source, &fmd);

  // no nulls and no repetition, so the only encoding used should be for the data.
  // since we're writing v1, both dict and data pages should use PLAIN_DICTIONARY.
  auto const expect_enc = [&fmd](int idx, cudf::io::parquet::detail::Encoding enc) {
    EXPECT_EQ(fmd.row_groups[0].columns[idx].meta_data.encodings[0], enc);
  };

  // requested plain
  expect_enc(0, Encoding::PLAIN);
  // requested dictionary
  expect_enc(1, Encoding::PLAIN_DICTIONARY);
  // requested delta_binary_packed
  expect_enc(2, Encoding::DELTA_BINARY_PACKED);
  // requested delta_length_byte_array, but should fall back to dictionary
  expect_enc(3, Encoding::PLAIN_DICTIONARY);
  // requested delta_byte_array, but should fall back to dictionary
  expect_enc(4, Encoding::PLAIN_DICTIONARY);
  // no request, should use dictionary
  expect_enc(5, Encoding::PLAIN_DICTIONARY);

  // requested plain
  expect_enc(6, Encoding::PLAIN);
  // requested dictionary
  expect_enc(7, Encoding::PLAIN_DICTIONARY);
  // requested delta_length_byte_array
  expect_enc(8, Encoding::DELTA_LENGTH_BYTE_ARRAY);
  // requested delta_byte_array
  expect_enc(9, Encoding::DELTA_BYTE_ARRAY);
  // requested delta_binary_packed, but should fall back to dictionary
  expect_enc(10, Encoding::PLAIN_DICTIONARY);
  // no request, should use dictionary
  expect_enc(11, Encoding::PLAIN_DICTIONARY);
}

TEST_F(ParquetWriterTest, Decimal128DeltaByteArray)
{
  // decimal128 in cuDF maps to FIXED_LEN_BYTE_ARRAY, which is allowed by the spec to use
  // DELTA_BYTE_ARRAY encoding. But this use is not implemented in cuDF.
  __int128_t val0 = 0xa1b2'c3d4'e5f6ULL;
  __int128_t val1 = val0 << 80;
  column_wrapper<numeric::decimal128> col0{{numeric::decimal128(val0, numeric::scale_type{0}),
                                            numeric::decimal128(val1, numeric::scale_type{0})}};

  auto expected = table_view{{col0, col0}};
  cudf::io::table_input_metadata table_metadata(expected);
  table_metadata.column_metadata[0]
    .set_name("decimal128")
    .set_encoding(cudf::io::column_encoding::DELTA_BYTE_ARRAY)
    .set_nullability(false);

  auto const filepath = temp_env->get_temp_filepath("Decimal128DeltaByteArray.parquet");
  const cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .compression(cudf::io::compression_type::NONE)
      .metadata(table_metadata);
  cudf::io::write_parquet(out_opts);

  auto const source = cudf::io::datasource::create(filepath);
  cudf::io::parquet::detail::FileMetaData fmd;
  read_footer(source, &fmd);

  // make sure DELTA_BYTE_ARRAY was not used
  EXPECT_NE(fmd.row_groups[0].columns[0].meta_data.encodings[0],
            cudf::io::parquet::detail::Encoding::DELTA_BYTE_ARRAY);
}

TEST_F(ParquetWriterTest, DeltaBinaryStartsWithNulls)
{
  // test that the DELTA_BINARY_PACKED writer can properly encode a column that begins with
  // more than 129 nulls
  constexpr int num_rows  = 500;
  constexpr int num_nulls = 150;

  auto const ones = thrust::make_constant_iterator(1);
  auto valids     = cudf::detail::make_counting_transform_iterator(
    0, [num_nulls](auto i) { return i >= num_nulls; });
  auto const col      = cudf::test::fixed_width_column_wrapper<int>{ones, ones + num_rows, valids};
  auto const expected = table_view({col});

  auto const filepath = temp_env->get_temp_filepath("DeltaBinaryStartsWithNulls.parquet");
  cudf::io::parquet_writer_options out_opts =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info{filepath}, expected)
      .write_v2_headers(true)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER);
  cudf::io::write_parquet(out_opts);

  cudf::io::parquet_reader_options in_opts =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info{filepath});
  auto result = cudf::io::read_parquet(in_opts);
  CUDF_TEST_EXPECT_TABLES_EQUAL(expected, result.tbl->view());
}

/////////////////////////////////////////////////////////////
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

////////////////////////
// Numeric type tests

// Typed test fixture for numeric type tests
template <typename T>
struct ParquetWriterNumericTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(ParquetWriterNumericTypeTest, SupportedTypes);

TYPED_TEST(ParquetWriterNumericTypeTest, SingleColumn)
{
  auto sequence =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return TypeParam(i % 400); });

  constexpr auto num_rows = 800;
  column_wrapper<TypeParam> col(sequence, sequence + num_rows, no_nulls());

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

/////////////////////////
// timestamp type tests

// Typed test fixture for timestamp type tests
template <typename T>
struct ParquetWriterTimestampTypeTest : public ParquetWriterTest {
  auto type() { return cudf::data_type{cudf::type_to_id<T>()}; }
};

TYPED_TEST_SUITE(ParquetWriterTimestampTypeTest, SupportedTimestampTypes);

TYPED_TEST(ParquetWriterTimestampTypeTest, Timestamps)
{
  auto sequence = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return ((std::rand() / 10000) * 1000); });

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, no_nulls());

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

  constexpr auto num_rows = 100;
  column_wrapper<TypeParam, typename decltype(sequence)::value_type> col(
    sequence, sequence + num_rows, no_nulls());
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

//////////////////////////////
// writer stress tests

// Base test fixture for "stress" tests
struct ParquetWriterStressTest : public cudf::test::BaseFixture {};

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
