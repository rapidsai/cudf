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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

class ORCTest : public cudf::test::BaseFixture {};

template <typename... UniqPtrs>
std::vector<std::unique_ptr<cudf::column>> make_uniqueptrs_vector(UniqPtrs&&... uniqptrs)
{
  std::vector<std::unique_ptr<cudf::column>> ptrsvec;
  (ptrsvec.push_back(std::forward<UniqPtrs>(uniqptrs)), ...);
  return ptrsvec;
}

cudf::table construct_table()
{
  constexpr auto num_rows = 10;

  auto const zeros_iterator = thrust::make_constant_iterator(0);
  auto const ones_iterator  = thrust::make_constant_iterator(1);

  cudf::test::fixed_width_column_wrapper<bool> col0(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_width_column_wrapper<int8_t> col1(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_width_column_wrapper<int16_t> col2(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_width_column_wrapper<int32_t> col3(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_width_column_wrapper<float> col4(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_width_column_wrapper<double> col5(zeros_iterator, zeros_iterator + num_rows);
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col6(
    ones_iterator, ones_iterator + num_rows, numeric::scale_type{12});
  cudf::test::fixed_point_column_wrapper<numeric::decimal128::rep> col7(
    ones_iterator, ones_iterator + num_rows, numeric::scale_type{-12});

  cudf::test::lists_column_wrapper<int64_t> col8 = [] {
    auto col8_mask =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });
    return cudf::test::lists_column_wrapper<int64_t>(
      {{1, 1}, {1, 1, 1}, {}, {1}, {1, 1, 1, 1}, {1, 1, 1, 1, 1}, {}, {1, -1}, {}, {-1, -1}},
      col8_mask);
  }();

  cudf::test::structs_column_wrapper col9 = [&ones_iterator] {
    auto child_col_mask =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i % 2); });
    cudf::test::fixed_width_column_wrapper<int32_t> child_col(
      ones_iterator, ones_iterator + num_rows, child_col_mask);
    return cudf::test::structs_column_wrapper{child_col};
  }();

  cudf::test::strings_column_wrapper col10 = [] {
    std::vector<std::string> col10_data(num_rows, "rapids");
    return cudf::test::strings_column_wrapper(col10_data.begin(), col10_data.end());
  }();

  auto colsptr = make_uniqueptrs_vector(col0.release(),
                                        col1.release(),
                                        col2.release(),
                                        col3.release(),
                                        col4.release(),
                                        col5.release(),
                                        col6.release(),
                                        col7.release(),
                                        col8.release(),
                                        col9.release(),
                                        col10.release());
  return cudf::table(std::move(colsptr));
}

TEST_F(ORCTest, ORCChunkedReader)
{
  auto tab      = construct_table();
  auto filepath = temp_env->get_temp_filepath("OrcMultiColumn.orc");
  cudf::io::orc_writer_options out_opts =
    cudf::io::orc_writer_options::builder(cudf::io::sink_info{filepath}, tab);
  cudf::io::write_orc(out_opts, cudf::test::get_default_stream());

  auto stream = cudf::test::get_default_stream();

  auto const read_opts =
    cudf::io::orc_reader_options::builder(cudf::io::source_info{filepath}).build();
  auto reader = cudf::io::chunked_orc_reader(static_cast<std::size_t>(0UL),
                                             static_cast<std::size_t>(1UL),
                                             static_cast<cudf::size_type>(10000),
                                             read_opts,
                                             stream);

  printf(" test line %d, stream = %zu\n", __LINE__, (size_t)stream.value());
  printf("line %d\n", __LINE__);
  fflush(stdout);

  auto out_tables = std::vector<std::unique_ptr<cudf::table>>{};

  do {
    auto chunk = reader.read_chunk();
    out_tables.emplace_back(std::move(chunk.tbl));
  } while (reader.has_next());
}
